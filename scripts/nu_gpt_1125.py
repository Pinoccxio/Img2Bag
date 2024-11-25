import os, argparse
import json
import numpy as np
import rosbag, rospy
import cv2
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from bisect import bisect_left


def parse_args():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='读取RosBag包消息')
    parser.add_argument('--bag', type=str, help='RosBag包路径', required=True)
    parser.add_argument('--output_dir', type=str, help='标注保存文件路径', required=True)
    return parser.parse_args()


def save_point_cloud_to_pcd(point_cloud_msg, pcd_file):
    points = []
    for point in pc2.read_points(point_cloud_msg, skip_nans=True):
        points.append([point[0], point[1], point[2]])
    points = np.array(points)
    # 保存为 PCD 文件
    with open(pcd_file, 'w') as f:
        f.write('# .PCD v0.7 - Point Cloud Data file format\n')
        f.write('VERSION 0.7\n')
        f.write('FIELDS x y z\n')
        f.write('SIZE 4 4 4\n')
        f.write('TYPE F F F\n')
        f.write('COUNT 1 1 1\n')
        f.write('WIDTH {}\n'.format(len(points)))
        f.write('HEIGHT 1\n')
        f.write('VIEWPOINT 0 0 0 1 0 0 0\n')
        f.write('POINTS {}\n'.format(len(points)))
        f.write('DATA ascii\n')
        np.savetxt(f, points, fmt='%f %f %f')


def save_image_to_jpg(image_msg, image_file, bridge):
    cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
    cv2.imwrite(image_file, cv_image)


def fill_obs_list(obstacle_world_coords_dict, json_file):
    with open(json_file, 'r', encoding='utf-8') as annotation_json:
        print("Reading Annotation.json")
        content = annotation_json.read()
        annotations = json.loads(content)
        for anno in annotations:
            pos = np.array([anno["x"], anno["y"], anno["z"]])
            cat = anno['category']
            label = anno['label']
            size = np.array([anno['w'], anno['l'], anno['h']])
            obstacle_world_coords_dict.append({"pos": pos, "cat": cat, "size": size, "label": label})
    print("Annotation.json Closed")


def is_in_view(pos_obs, K_cam, H, W):
    # Obstacle coords
    z = pos_obs[2]
    x = pos_obs[0]/z
    y = pos_obs[1]/z

    # Pixel
    u = K_cam[0, 0] * x + K_cam[0, 2]
    v = K_cam[1, 1] * y + K_cam[1, 2]

    # Behind the camera
    if z <= 0:
        return False

    # In the screen
    if 0 <= u < H and 0 <= v < W:
        return True
    return False


def calculate_visibility(obs_bbox, H, W):
    """
    计算标注框的可见度
    obs_bbox: 标注框的坐标（左上角和右下角）
    H: 图像的高度
    W: 图像的宽度
    返回：可见度百分比
    """
    x_min, y_min, x_max, y_max = obs_bbox
    
    # 计算标注框在图像内的可见区域
    visible_x_min = max(0, x_min)
    visible_y_min = max(0, y_min)
    visible_x_max = min(W, x_max)
    visible_y_max = min(H, y_max)

    # 如果标注框完全在图像外，则可见度为0
    if visible_x_min >= visible_x_max or visible_y_min >= visible_y_max:
        return 0

    # 计算标注框可见部分的面积与原标注框面积的比例
    visible_area = (visible_x_max - visible_x_min) * (visible_y_max - visible_y_min)
    total_area = (x_max - x_min) * (y_max - y_min)

    visibility = visible_area / total_area
    # 根据可见度，返回百分比
    if visibility <= 0.4:
        return 1
    elif visibility <= 0.6:
        return 2
    elif visibility <= 0.8:
        return 3
    else:
        return 4


def process_rosbag(bag_file, output_dir):
    bridge = CvBridge()
    bag = rosbag.Bag(bag_file, 'r')
    bag_name = os.path.basename(bag_file)
    scene_name = os.path.splitext(bag_name)[0]

    os.makedirs(os.path.join(output_dir, 'sweeps/LIDAR_TOP'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'sweeps/CAM_FRONT'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'samples/LIDAR_TOP'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'samples/CAM_FRONT'), exist_ok=True)

    scenes = []
    samples = []
    sample_datas = []
    sample_annotations = []
    ego_poses = []
    calibrated_sensors = []
    
    sensors = []
    logs = []

    obstacle_world_coords_dict = []
    anno_file = '/home/cx/dataset/annotations.json'
    fill_obs_list(obstacle_world_coords_dict, anno_file)

# ===================================== Transform Matrix =====================================
    # World2Body
    T_World2Body = np.eye(4)
    t_World2Body = np.array([20, 20, 6.2])
    # T_World2Body[:3, 3] = t_World2Body

    # Body2Lidar
    T_Body2Lidar = np.eye(4)
    t_Body2Lidar = np.array([0.045, 0, 0.18])
    T_Body2Lidar[:3, 3] = t_Body2Lidar

    # Cam2Lidar -> Lidar2Cam
    T_Cam2Lidar = np.eye(4)
    t_Cam2Lidar = np.array([-0.105, 0, 0.14])
    T_Cam2Lidar[:3, 3] = t_Cam2Lidar
    T_Lidar2Cam = np.linalg.inv(T_Cam2Lidar)

    # Camera Intrinsic
    K = np.array([[958.8, 0, 957.8], [0, 956.7, 589.5], [0, 0, 1]])
# ===================================== Transform Matrix =====================================



# ===================================== Scene & Log =====================================

    scene_token = f"scene_token_{scene_name}"
    log_token = f"log_token_{scene_name}"
    vehicle = 'Four-Feet Robot'
    logfile = f'{scene_name}_log'

    scene = {
        'token': scene_token,
        'name': scene_name,
        'description': 'Moon Example',
        'log_token': log_token,
        'nbr_samples': 0,  # 之后更新
        'first_sample_token': '',  # 之后更新
        'last_sample_token': '',   # 之后更新
    }
    scenes.append(scene)

    log = {
        'token': log_token,
        'vehicle': vehicle,
        'date_captured': scene_name,
        'location': 'Issac Sim',
        'logfile': logfile,
        'vehicle_name': 'Sim Car',
        'scene_token': scene_token,
    }
    logs.append(log)

# ===================================== Scene & Log =====================================

# ======================================== Sensor ========================================

    # Sensor Info
    sensor_token_camera = "sensor_token_camera"
    sensor_token_lidar = "sensor_token_lidar"

    # >>> sensor.json >>>
    sensor_camera = {
        'token': sensor_token_camera,
        'channel': 'CAM_FRONT',
        'modality': 'camera',
    }
    sensor_lidar = {
        'token': sensor_token_lidar,
        'channel': 'LIDAR_TOP',
        'modality': 'lidar',
    }
    sensors.extend([sensor_camera, sensor_lidar])
    # <<< sensor <<<

    # >>> calibrated_sensors,json >>>
    calibrated_sensor_camera = {
        'token': sensor_token_camera,
        'sensor_token': sensor_token_camera,
        'translation': [20.15, 20, 6.52],  # 自行设置
        'rotation': [0, 0, 0, 1],  # 单位四元数
        'camera_intrinsic': [[958.8, 0, 957.8], [0, 956.7, 589.5], [0, 0, 1]],
    }
    calibrated_sensor_lidar = {
        'token': sensor_token_lidar,
        'sensor_token': sensor_token_lidar,
        'translation': [20.045, 20, 6.38],
        'rotation': [0, 0, 0, 1],
        'camera_intrinsic': []
    }
    calibrated_sensors.extend([calibrated_sensor_camera, calibrated_sensor_lidar])
    # <<< calibrated_sensors <<<

# ======================================== Sensor ========================================

    # Index and Cache
    odom_cache = []
    rgb_cache = []
    odom_timestamps = []
    rgb_timestamps = []

    print("Indexing /odom and /rgb_data messages...")
    for topic, msg, t in tqdm(bag.read_messages(topics=['/odom', '/rgb_data'])):
        timestamp = t.to_sec()
        if topic == '/odom':
            odom_cache.append((timestamp, msg))
            odom_timestamps.append(timestamp)
        elif topic == '/rgb_data':
            rgb_cache.append((timestamp, msg))
            rgb_timestamps.append(timestamp)

    odom_timestamps, odom_cache = zip(*sorted(zip(odom_timestamps, odom_cache)))
    rgb_timestamps, rgb_cache = zip(*sorted(zip(rgb_timestamps, rgb_cache)))

    odom_timestamps = list(odom_timestamps)
    rgb_timestamps = list(rgb_timestamps)
    sample_idx = 0
    print("Processing /pc_scan messages...")
    for idx, (topic, pc_msg, t) in enumerate(tqdm(bag.read_messages(topics=['/pc_scan']))):
    
        pc_timestamp = t.to_sec()
        is_key_frame = False
        sample_token = f"sample_token_{sample_idx:06d}"
        
        # 2Hz key frame
        if (idx % 5 == 0):
            is_key_frame = True
            sample_idx = sample_idx + 1

        # 根据pcd的timestamp获取最近的odom和rgb数据
        odom_index = bisect_left(odom_timestamps, pc_timestamp)
        rgb_index = bisect_left(rgb_timestamps, pc_timestamp)

        odom_msg = odom_cache[odom_index - 1][1] if odom_index > 0 else odom_cache[0][1]
        odom_timstamp = odom_cache[odom_index - 1][0] if odom_index > 0 else odom_cache[0][1]
        rgb_msg = rgb_cache[rgb_index - 1][1] if rgb_index > 0 else rgb_cache[0][1]

# ======================================== ego_pose ========================================

        ego_pose_token = f"ego_pose_token_{idx:06d}"
        odom_orientation = odom_msg.pose.pose.orientation
        odom_position = odom_msg.pose.pose.position
        ego_pose = {
            'token': ego_pose_token,
            'timestamp': int(odom_timstamp * 1e6),
            'rotation': [
                odom_orientation.w,
                odom_orientation.x,
                odom_orientation.y,
                odom_orientation.z
            ],
            'translation': [
                odom_position.x,
                odom_position.y,
                odom_position.z
            ],
        }
        ego_poses.append(ego_pose)
# ======================================== ego_pose ========================================

        # >>> Save pcd and jpg >>>
        pc_filename = f"sweeps/LIDAR_TOP/pc_{int(pc_timestamp * 1e6)}.pcd"
        pc_filepath = os.path.join(output_dir, pc_filename)
        save_point_cloud_to_pcd(pc_msg, pc_filepath)
        
        img_filename = f"sweeps/CAM_FRONT/img_{int(pc_timestamp * 1e6)}.jpg"
        img_filepath = os.path.join(output_dir, img_filename)
        save_image_to_jpg(rgb_msg, img_filepath, bridge)
        # <<< Save pcd and jpg <<<

# ====================================== sample_data ======================================
        sd_token_lidar = f"sample_data_token_{(idx*2):06d}"
        sd_lidar = {
            'token': sd_token_lidar,
            'sample_token': sample_token,
            'ego_pose_token': ego_pose_token,
            'calibrated_sensor_token': sensor_token_lidar,
            'timestamp': int(pc_timestamp * 1e6),
            'fileformat': 'pcd',
            'is_key_frame': is_key_frame,
            'height': 0,
            'width': 0,
            'filename': pc_filename,
            'prev': '',
            'next': '',
        }
        sample_datas.append(sd_lidar)

        sd_token_camera = f"sample_data_token_{(idx*2 + 1):06d}"
        sd_camera = {
            'token': sd_token_camera,
            'sample_token': sample_token,
            'ego_pose_token': ego_pose_token,
            'calibrated_sensor_token': sensor_token_camera,
            'timestamp': int(pc_timestamp * 1e6),
            'fileformat': 'jpg',
            'is_key_frame': is_key_frame,
            'height': rgb_msg.height,
            'width': rgb_msg.width,
            'filename': img_filename,
            'prev': '',
            'next': '',
        }
        sample_datas.append(sd_camera)
# ====================================== sample_data ======================================

# ======================================== sample ========================================
        if is_key_frame:
            print(f"Saving key-frame data at {int(pc_timestamp * 1e6)}\n")
            os.makedirs(os.path.join(output_dir, 'samples/LIDAR_TOP'), exist_ok=True)
            pc_filename_sample = f"samples/LIDAR_TOP/pc_{int(pc_timestamp * 1e6)}.pcd"
            pc_filepath_sample = os.path.join(output_dir, pc_filename_sample)
            save_point_cloud_to_pcd(pc_msg, pc_filepath_sample)

            os.makedirs(os.path.join(output_dir, 'samples/CAM_FRONT'), exist_ok=True)
            img_filename_sample = f"samples/CAM_FRONT/img_{int(pc_timestamp * 1e6)}.jpg"
            img_filepath_sample = os.path.join(output_dir, img_filename_sample)
            save_image_to_jpg(rgb_msg, img_filepath_sample, bridge)

            # 创建sample条目
            sample = {
                'token': sample_token,
                'timestamp': int(pc_timestamp * 1e6),
                'prev': '',
                'next': '',
                'scene_token': scene_token,
            }
            samples.append(sample)
# ======================================== sample ========================================

# ======================================== sample_annotation ========================================
            quaternion = [odom_orientation.x, odom_orientation.y, odom_orientation.z, odom_orientation.w]
            rotation = R.from_quat(quaternion).as_matrix()
            translation = [odom_position.x, odom_position.y, odom_position.z]
            T_World2Body[:3, 3] = t_World2Body + translation
            T_World2Body[:3, :3] = rotation
            T_World2Cam = np.dot(T_Lidar2Cam, np.dot(T_Body2Lidar, T_World2Body))
            obs_count = 0
            for obs in obstacle_world_coords_dict:
                    pos_obs = obs['pos']
                    pos_obs = [pos_obs[0], pos_obs[1], pos_obs[2], 1]
                    size_obs = obs['size'].tolist()
                    pos_cam = np.dot(T_World2Cam, pos_obs)
                    if (is_in_view(pos_cam, K, rgb_msg.height, rgb_msg.width)):
                        sample_anno_token = f'sample_annotation_{obs_count:06d}'
                        prev_token = sample_annotations[obs_count - 1]['token'] if obs_count > 0 else ''  
                        next_token = ''  # 默认为空，后续将在列表末尾添加  
                        obs_count = obs_count + 1
                        # >>> bbox_coords >>>
                        x_min, y_min = pos_cam[0] - size_obs[0] / 2, pos_cam[1] - size_obs[1] / 2
                        x_max, y_max = pos_cam[0] + size_obs[0] / 2, pos_cam[1] + size_obs[1] / 2
                        # <<< bbox_coords <<<
                        visibility = calculate_visibility([x_min, y_min, x_max, y_max], rgb_msg.height, rgb_msg.width)
                        label = obs['label']
                        num_lidar_pts = 0
                        for point in pc2.read_points(pc_msg, skip_nans=True):
                            pcd_homo = [point[0], point[1], point[2], 1]
                            pcd_cam = np.dot(T_Lidar2Cam, pcd_homo)
                            if is_in_view(pcd_cam, K, rgb_msg.height, rgb_msg.width):
                                if x_min <= pcd_cam[0] <= x_max and y_min <= pcd_cam[1] <= y_max:
                                    num_lidar_pts = num_lidar_pts + 1
                        print(f"{num_lidar_pts} lidar points in label-{label}")
                        sample_annotation = {
                            "token": sample_anno_token,  
                            "sample_token": sample_token,
                            "visibility_token": f"{visibility}",  
                            "attribute_tokens": [
                                f"attribute_token_000{label}"
                            ],
                            "translation": translation,  
                            "size": size_obs,  
                            "rotation": quaternion,
                            "prev": prev_token,  
                            "next": next_token,
                            "num_lidar_pts": num_lidar_pts,
                            "num_radar_pts": 0  
                        }
                        sample_annotations.append(sample_annotation)
            # 更新每个字典中的next字段  
            for i in range(len(sample_annotations) - 1):  
                sample_annotations[i]['next'] = sample_annotations[i + 1]['token']  
# ======================================== sample_annotation ========================================

    # Update Scene 
    scenes[0]['first_sample_token'] = samples[0]['token']
    scenes[0]['last_sample_token'] = samples[-1]['token']
    scenes[0]['nbr_samples'] = len(samples)

# ======================================== JSON ========================================
    metadata = {
        'scene': scenes,
        'sample': samples,
        'sample_data': sample_datas,
        'sample_annotation': sample_annotations,
        'ego_pose': ego_poses,
        'calibrated_sensor': calibrated_sensors,
        'sensor': sensors,
        'log': logs,
    }
    print("Ready to write json files")

    for key, value in metadata.items():
        with open(os.path.join(output_dir, f'{key}.json'), 'w') as f:
            json.dump(value, f, indent=4)
        print(f"Saved into {key}.json")
# ======================================== JSON ========================================
    bag.close()


if __name__ == "__main__":
    args = parse_args()
    process_rosbag(args.bag, args.output_dir)
