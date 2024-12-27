import os, argparse
import json
import numpy as np
import rosbag, rospy
import cv2
import struct
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from bisect import bisect_left


# python nuscenes_anno.py --bag /home/cx/dataset/isaac_sim/isaac_bags/2024-09-30-09-10-54.bag --output_dir /home/cx/dataset/isaac_sim/dataset/0930_0910 --tag normal

def parse_args():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='读取RosBag包消息')
    parser.add_argument('--tag', type=str, help='开放场景or建筑', default='open')
    parser.add_argument('--bag', type=str, help='RosBag包路径', required=True)
    parser.add_argument('--output_dir', type=str, help='标注保存文件路径', required=True)
    return parser.parse_args()


def calculate_ring_index(point):
    x, y, z = point
    # 激光雷达的垂直视场范围：从 -15° 到 +15°，总共 30°
    min_angle = -15  # 最小角度
    max_angle = 15   # 最大角度
    num_rings = 32    # 32个环
    
    # 计算垂直角度：通过 arctan2(z, r) 来计算
    # 假设 r 为点到激光雷达的水平距离，可以通过 sqrt(x^2 + y^2) 来获得
    r = np.sqrt(x**2 + y**2)
    angle = np.arctan2(z, r) * 180 / np.pi  # 将 z 值和 r 值转换为角度（单位：度）

    # 限制角度在 min_angle 和 max_angle 范围内
    if angle < min_angle:
        angle = min_angle
    elif angle > max_angle:
        angle = max_angle
    
    # 每个环之间的角度差
    angle_step = (max_angle - min_angle) / (num_rings - 1)

    # 计算该角度对应的环索引
    ring_index = round((angle - min_angle) / angle_step)
    
    return ring_index


def save_point_cloud_to_pcd(point_cloud_msg, bin_file):  
    bin_data = b''
    point_cloud_data = list(pc2.read_points(point_cloud_msg, field_names=("x", "y", "z"), skip_nans=True))
    for point in point_cloud_data:
            x, y, z = point
            ring = float(calculate_ring_index(point))
            intensity = 100.0
            bin_data += struct.pack('fffff', -y, x, z, intensity, ring)

    with open(bin_file, 'wb') as f:
        f.write(bin_data)


def save_image_to_jpg(image_msg, image_file, bridge):
    cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
    cv2.imwrite(image_file, cv_image)


def fill_obs_list(json_file):
    obstacle_world_coords_dict = []
    with open(json_file, 'r', encoding='utf-8') as annotation_json:
        print("Reading Annotation.json")
        content = annotation_json.read()
        annotations = json.loads(content)
        for anno in annotations:
            pos = np.array([anno["x"], anno["y"], anno["z"]])
            cat = anno['category']
            label = anno['label']
            size = np.array([anno['w'], anno['l'], anno['h']])
            anno_id = anno['id']
            obstacle_world_coords_dict.append({"pos": pos, "category": cat, "size": size, "label": label, "id": anno_id})
    return obstacle_world_coords_dict


def num_pts_add(point, pos_obs, size_obs):
    x, y, z = point
    x_min, y_min, z_min = pos_obs[0] - size_obs[0] / 2, pos_obs[1] - size_obs[1] / 2, pos_obs[2] - size_obs[2] / 2
    x_max, y_max, z_max = pos_obs[0] + size_obs[0] / 2, pos_obs[1] + size_obs[1] / 2, pos_obs[2] + size_obs[2] / 2
    if (x <= x_max and x >= x_min) and (y <= y_max and y >= y_min) and (z <= z_max and z >= z_min):
        return True
    else:
        return False

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


def transform_to_camera_coordinates(world_coords, T_World2Cam):
    # 使用转换矩阵将世界坐标转换到相机坐标系
    world_coords_homogeneous = np.array(world_coords).reshape(4, 1)
    camera_coords_homogeneous = np.dot(T_World2Cam, world_coords_homogeneous)
    
    # 获取相机坐标
    x_cam, y_cam, z_cam = camera_coords_homogeneous[:3].flatten()
    
    return x_cam, y_cam, z_cam


def calculate_visibility(obs_bbox, K_cam, H, W, T_World2Cam):
    x_min, y_min, z_min, x_max, y_max, z_max = obs_bbox
    
    # 转换标注框的世界坐标为相机坐标系下的坐标
    world_coords_min = [x_min, y_min, z_min, 1.0]  # 将 (x_min, y_min, z_min) 放到世界坐标中
    world_coords_max = [x_max, y_max, z_max, 1.0]  # 同理
    
    # 获取相机坐标系下的坐标
    # x_min_cam, y_min_cam, z_min_cam = transform_to_camera_coordinates(world_coords_min, T_World2Cam)
    # x_max_cam, y_max_cam, z_max_cam = transform_to_camera_coordinates(world_coords_max, T_World2Cam)
    x_min_cam, y_min_cam, z_min_cam = np.dot(T_World2Cam, world_coords_min)[:3].flatten()
    x_max_cam, y_max_cam, z_max_cam = np.dot(T_World2Cam, world_coords_max)[:3].flatten()

    # 根据相机坐标计算像素坐标
    u_min = K_cam[0, 0] * x_min_cam / z_min_cam + K_cam[0, 2]
    v_min = K_cam[1, 1] * y_min_cam / z_min_cam + K_cam[1, 2]
    
    u_max = K_cam[0, 0] * x_max_cam / z_max_cam + K_cam[0, 2]
    v_max = K_cam[1, 1] * y_max_cam / z_max_cam + K_cam[1, 2]
    
    # 计算标注框在图像内的可见区域
    visible_x_min = max(0, u_min)
    visible_y_min = max(0, v_min)
    visible_x_max = min(W, u_max)
    visible_y_max = min(H, v_max)

    # 如果标注框完全在图像外，则可见度为0
    if visible_x_min >= visible_x_max or visible_y_min >= visible_y_max:
        return 0

    # 计算标注框可见部分的面积与原标注框面积的比例
    visible_area = (visible_x_max - visible_x_min) * (visible_y_max - visible_y_min)
    total_area = (u_max - u_min) * (v_max - v_min)

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


def process_rosbag(bag_file, output_dir, tag):
    bridge = CvBridge() 
    bag = rosbag.Bag(bag_file, 'r')
    bag_name = os.path.basename(bag_file)
    scene_time = os.path.splitext(bag_name)[0]
    scene_name = str(scene_time).split('-')[1] + str(scene_time).split('-')[2] + '-' + str(scene_time).split('-')[3] + str(scene_time).split('-')[4]

    sweep_lidar_folder = os.path.join(output_dir, 'sweeps/LIDAR_TOP') 
    # os.makedirs(sweep_lidar_folder, exist_ok=True)

    sweep_camera_folder = os.path.join(output_dir, 'sweeps/CAM_FRONT') 
    # os.makedirs(sweep_camera_folder, exist_ok=True)

    sample_lidar_folder = os.path.join(output_dir, 'samples/LIDAR_TOP') 
    # os.makedirs(sample_lidar_folder, exist_ok=True)
    
    sample_camera_folder = os.path.join(output_dir, 'samples/CAM_FRONT') 
    # os.makedirs(sample_camera_folder, exist_ok=True)

    scenes = []
    samples = []
    sample_datas = []
    sample_annotations = []
    ego_poses = []
    calibrated_sensors = []
    instance_list = []
    
    sensors = []
    logs = []

    obstacle_world_coords_dict =[]
    
    anno_normal_file = '/home/cx/dataset/isaac_sim/annotations_normal.json'
    anno_open_file = '/home/cx/dataset/isaac_sim/annotations_open.json'

    if tag == 'normal':
        print(f'Reading in a NORMAL situation')
        t_World2Body = np.array([20, 20, 6.2])
        obstacle_world_coords_dict = fill_obs_list(anno_normal_file)
        calibrated_sensor_token_camera = "sensor_token_camera_normal"
        calibrated_sensor_token_lidar = "sensor_token_lidar_normal"
    else:
        print(f'Reading in a OEPN situation')
        t_World2Body = np.array([20, 20, 0.1])
        obstacle_world_coords_dict = fill_obs_list(anno_open_file)
        calibrated_sensor_token_camera = "sensor_token_camera_open"
        calibrated_sensor_token_lidar = "sensor_token_lidar_open"
    

# ===================================== Transform Matrix =====================================
    # World2Body
    T_World2Body = np.eye(4)

    # Body2Lidar
    T_Body2Lidar = np.eye(4)
    t_Body2Lidar = np.array([0.045, 0, 0.18])
    T_Body2Lidar[:3, 3] = t_Body2Lidar

    # Cam2Lidar -> Lidar2Cam
    T_Cam2Lidar = np.eye(4)
    r_Cam2Lidar = np.array([
        [ 0,  0,  1],   # Xl ->  Zc
        [-1,  0,  0],   # Yl -> -Xc
        [ 0, -1,  0]    # Zl -> -Yc
    ])
    t_Cam2Lidar = np.array([-0.105, 0, 0.14])
    T_Cam2Lidar[:3, :3] = r_Cam2Lidar
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

    cam_translation = t_Body2Lidar - t_Cam2Lidar
    lidar_translation = t_Body2Lidar

    # >>> calibrated_sensors,json >>>
    calibrated_sensor_camera = {
        'token': calibrated_sensor_token_camera,
        'sensor_token': sensor_token_camera,
        'translation': cam_translation.tolist(),  # 自行设置
        'rotation': [1, 0, 0, 0],  # 单位四元数
        'camera_intrinsic': [[958.8, 0, 957.8], [0, 956.7, 589.5], [0, 0, 1]],
    }
    calibrated_sensor_lidar = {
        'token': calibrated_sensor_token_lidar,
        'sensor_token': sensor_token_lidar,
        'translation': lidar_translation.tolist(),
        'rotation': [1, 0, 0, 0],
        'camera_intrinsic': []
    }
    calibrated_sensors.extend([calibrated_sensor_camera, calibrated_sensor_lidar])
    # <<< calibrated_sensors <<<

# ======================================== Sensor ========================================

# ==================================== Index and Cache ====================================
    odom_cache = []
    rgb_cache = []
    odom_timestamps = []
    rgb_timestamps = []

    print("Processing /odom messages...")
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

# ==================================== Index and Cache ====================================

    sample_idx = 0
    obs_count = 0
    sd_count = 0
    print("Processing /pc_scan messages...")
    for idx, (topic, pc_msg, t) in enumerate(tqdm(bag.read_messages(topics=['/pc_scan']))):
    
        pc_timestamp = t.to_sec()
        is_key_frame = False
        
        # 2Hz key frame
        if idx == 0:
            is_key_frame = True
            
        if (idx % 5 == 0) and (idx != 0):
            is_key_frame = True
            sample_idx = sample_idx + 1


        sample_token = f"sample_token_{scene_name}_{sample_idx:06d}"

        # 根据pcd的timestamp获取最近的odom和rgb数据
        odom_index = bisect_left(odom_timestamps, pc_timestamp)
        rgb_index = bisect_left(rgb_timestamps, pc_timestamp)
        
        odom_msg = odom_cache[odom_index - 1][1] if odom_index > 0 else odom_cache[0][1]
        odom_timstamp = odom_cache[odom_index - 1][0] if odom_index > 0 else odom_cache[0][1]
        rgb_msg = rgb_cache[rgb_index - 1][1] if rgb_index > 0 else rgb_cache[0][1]

# ======================================== ego_pose ========================================
        ego_pose_token = f"ego_token_{scene_name}_{idx:06d}"
        odom_orientation = odom_msg.pose.pose.orientation
        odom_position = odom_msg.pose.pose.position + t_World2Body
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
        
        # >>> Save pcd >>>
        sweep_bin_file = f"pc_{int(pc_timestamp * 1e6)}.pcd.bin"
        sweep_bin_path = os.path.join(sweep_lidar_folder, sweep_bin_file)
        # save_point_cloud_to_pcd(pc_msg, sweep_bin_path)
        # <<< Save pcd <<<

# ================================================= LIDAR =================================================

# ====================================== lidar_sample_data ======================================
        sd_token_lidar = f"sd_token_{scene_name}_{(idx):06d}"
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
            'filename': f"sweeps/LIDAR_TOP/{sweep_bin_file}",
            'prev': '',
            'next': '',
        }
        sample_datas.append(sd_lidar)
        sd_count += 1
# ====================================== lidar_sample_data ======================================

# ======================================== sample ========================================
        if is_key_frame:
            sample_bin_file = f"pc_{int(pc_timestamp * 1e6)}.pcd.bin"
            sample_bin_path = os.path.join(sample_lidar_folder, sample_bin_file)
            # save_point_cloud_to_pcd(pc_msg, sample_bin_path)

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
            
            for obs in obstacle_world_coords_dict:
                    pos_obs = obs['pos']
                    pos_obs_homo = [pos_obs[0], pos_obs[1], pos_obs[2], 1]
                    pos_cam = np.dot(T_World2Cam, pos_obs_homo)
                    size_obs = obs['size'].tolist()
                    
                    # >>> bbox_coords >>>
                    x_min, y_min, z_min = pos_obs[0] - size_obs[0] / 2, pos_obs[1] - size_obs[1] / 2, pos_obs[2] - size_obs[2] / 2
                    x_max, y_max, z_max = pos_obs[0] + size_obs[0] / 2, pos_obs[1] + size_obs[1] / 2, pos_obs[2] + size_obs[2] / 2
                    # <<< bbox_coords <<<                 
                        
                    if (is_in_view(pos_cam, K, rgb_msg.height, rgb_msg.width)):
                        sample_anno_token = f'sa_token_{scene_name}_{obs_count:06d}'
                        prev_token = sample_annotations[obs_count - 1]['token'] if obs_count > 0 else ''  
                        next_token = ''  # 默认为空，后续将在列表末尾添加  
                        obs_count = obs_count + 1
                        obs_id = obs["id"]

                        visibility = calculate_visibility([x_min, y_min, z_min, x_max, y_max, z_max], K, rgb_msg.height, rgb_msg.width, T_World2Cam)
                        print(f"visibility level: v-{visibility}")
                        
                        label = obs['label']

                        num_lidar_pts = 0
                        for point in pc2.read_points(pc_msg, skip_nans=True):
                            if num_pts_add(point, pos_obs, size_obs):
                                num_lidar_pts += 1
                        # print(f"{num_lidar_pts} lidar points in label-{label}\n")
                        
                        instance_token = f"instance_token_{scene_name}_{obs_id:06d}"
                        rot_obs = np.array([1,0,0,0]).tolist()
                        
                        sample_annotation = {
                            "token": sample_anno_token,  
                            "sample_token": sample_token,
                            "instance_token": instance_token,
                            "visibility_token": f"{visibility}",  
                            "attribute_tokens": [
                                f"attribute_token_00000{label}"
                            ],
                            "translation": pos_obs.tolist(),  
                            "size": size_obs,  
                            "rotation": rot_obs,
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


# ================================================= LIDAR =================================================


# ============================================= instance =============================================
            instance_data = {}
            sorted_sample_annotations = sorted(sample_annotations, key=lambda x: x['token'])
            for sample_annotation in sorted_sample_annotations:
                instance_token = sample_annotation['instance_token']
                sample_annotation_token = sample_annotation['token']
                cat_id = int(sample_annotation['attribute_tokens'][0].split('_')[-1])
                category_token = f"category_token_{cat_id:06d}"

                if instance_token not in instance_data:
                    instance_data[instance_token] = {
                        "token": instance_token,
                        "category_token": category_token,  # 可根据实际需求设置类别标识符
                        "nbr_annotations": 0,
                        "first_annotation_token": sample_annotation_token,
                        "last_annotation_token": sample_annotation_token
                    }

                # 更新 last_annotation_token
                instance_data[instance_token]["last_annotation_token"] = sample_annotation_token
                instance_data[instance_token]["nbr_annotations"] = instance_data[instance_token]["nbr_annotations"] + 1

            instance_list = list(instance_data.values())
# ============================================= instance =============================================


# ================================================= IMG =================================================
    obs_count = 0
    sample_idx_rgb = 0
    print("Processing /rgb_data messages...")
    for rgb_idx, (topic, rgb_msg, t) in enumerate(tqdm(bag.read_messages(topics=['/rgb_data']))):
    
        img_timestamp = t.to_sec()
        is_key_frame = False
        

        # 2Hz key frame
        if rgb_idx == 0:
            is_key_frame = True

        if (rgb_idx % 5 == 0) and (rgb_idx != 0):
            is_key_frame = True
            sample_idx_rgb = sample_idx_rgb + 1

        sample_token = f"sample_token_{scene_name}_{sample_idx_rgb:06d}"

        # >>> Save jpg >>>
        sweep_cam_file = f"img_{int(img_timestamp * 1e6)}.jpg"
        sweep_cam_path = os.path.join(sweep_camera_folder, sweep_cam_file)
        # save_image_to_jpg(rgb_msg, sweep_cam_path, bridge)
        # <<< Save jpg <<<

# ====================================== sample_data ======================================

        sd_token_camera = f"sd_token_{scene_name}_{(sd_count):06d}"
        ego_pose_token = f"ego_token_{scene_name}_{rgb_idx:06d}"
        sd_camera = {
            'token': sd_token_camera,
            'sample_token': sample_token,
            'ego_pose_token': ego_pose_token,
            'calibrated_sensor_token': sensor_token_camera,
            'timestamp': int(img_timestamp * 1e6),
            'fileformat': 'jpg',
            'is_key_frame': is_key_frame,
            'height': rgb_msg.height,
            'width': rgb_msg.width,
            'filename': f"sweeps/CAM_FRONT/{sweep_cam_file}",
            'prev': '',
            'next': '',
        }
        sample_datas.append(sd_camera)
        sd_count += 1
# ====================================== sample_data ======================================

        if is_key_frame:
            sample_cam_file = f"img_{int(img_timestamp * 1e6)}.jpg"
            sample_cam_path = os.path.join(sample_camera_folder, sample_cam_file)
            # save_image_to_jpg(rgb_msg, sample_cam_path, bridge)
            print(f"Saving key-frame data at {int(img_timestamp * 1e6)}\n")

# ================================================= IMG =================================================

# =============================== prev & next in sample/sample_data ===============================                
    for i in range(len(samples) - 1):
        samples[i]["prev"] = samples[i - 1]["token"] if i > 0 else ''
        samples[i]["next"] = samples[i + 1]["token"]    
    samples[len(samples) - 1]["prev"] = samples[len(samples) - 2]["token"]
    for i in range(len(sample_datas) - 1):
        sample_datas[i]["prev"] = sample_datas[i - 1]["token"] if i > 0 else ''
        sample_datas[i]["next"] = sample_datas[i + 1]["token"]    
    sample_datas[len(sample_datas) - 1]["prev"] = sample_datas[len(sample_datas) - 2]["token"]
# =============================== prev & next in sample/sample_data ===============================

    # Update Scene 
    scenes[0]['first_sample_token'] = samples[0]['token']
    scenes[0]['last_sample_token'] = samples[-1]['token']
    scenes[0]['nbr_samples'] = len(samples)

# ============================================== JSON ==============================================
    metadata = {
        'scene': scenes,
        'sample': samples,
        'sample_data': sample_datas,
        'sample_annotation': sample_annotations,
        'ego_pose': ego_poses,
        'calibrated_sensor': calibrated_sensors,
        'sensor': sensors,
        'log': logs,
        'instance': instance_list
    }
    print("Ready to write json files")

    v_folder = 'v1.0-mini'
    v_path = os.path.join(output_dir, v_folder)
    os.makedirs(v_path, exist_ok=True)
    for key, value in metadata.items():
        with open(os.path.join(v_path, f'{key}.json'), 'w') as f:
            json.dump(value, f, indent=4)
        print(f"Saved into {key}.json")
# ======================================== JSON ========================================
    bag.close()


if __name__ == "__main__":
    args = parse_args()
    process_rosbag(args.bag, args.output_dir, args.tag)
