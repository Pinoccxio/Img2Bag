import os, argparse
import json
from typing import Union, Tuple, List
import numpy as np
import rosbag, rospy
import cv2
import struct
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
from pyquaternion import Quaternion
from tqdm import tqdm
from bisect import bisect_left
from nuscenes.utils import geometry_utils, data_classes
from shapely.geometry import MultiPoint, box

# python nuscenes_anno_1.py --bag /home/cx/dataset/isaac_sim/isaac_bags/2024-09-30-09-10-54.bag --output_dir /home/cx/dataset/isaac_sim/dataset/0930_0910 --tag normal

def parse_args():
    parser = argparse.ArgumentParser(description='读取RosBag包消息')
    parser.add_argument('--bag', type=str  , help='RosBag包路径', required=True)
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


def save_point_cloud_to_pcd(point_cloud_data, bin_file):  
    bin_data = b''
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
            # size = np.array([anno['w'], anno['l'], anno['h']])
            # size = np.array([anno['width'], anno['length'], anno['height']])
            size = np.array([0.3,0.3,0.6])
            anno_id = anno['id']
            obstacle_world_coords_dict.append({"pos": pos.tolist(), "category": cat, "size": size.tolist(), "label": label, "id": anno_id})
        # # 保存为 JSON 格式
        # with open('/home/cx/dataset/isaac_sim/dataset/info_obs.json', 'w', encoding='utf-8') as outfile:
        #     json.dump(obstacle_world_coords_dict, outfile, indent=4)
    return obstacle_world_coords_dict


def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1920, 1200)
) -> Union[np.ndarray, None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords]
        )

        # min_x = min(intersection_coords[:, 0])
        # min_y = min(intersection_coords[:, 1])
        # max_x = max(intersection_coords[:, 0])
        # max_y = max(intersection_coords[:, 1])

        return intersection_coords
    else:
        return None


def calculate_visibility(box_cam, K, image_shape):
    H, W = image_shape
    corners_3d = box_cam.corners()
    in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
    corners_3d = corners_3d[:, in_front]
    corner_coords = (geometry_utils.view_points(corners_3d, K, True).T[:, :2].tolist())
    final_coords = post_process_coords(corner_coords, (W, H))

    if final_coords is None:
        return 5
    else:
        full_area = ConvexHull(corner_coords).area
        real_area = ConvexHull(final_coords).area
        visibility = real_area/full_area
    
    if visibility <= 0.4 and visibility >= 0:
        return 1
    elif visibility <= 0.6:
        return 2
    elif visibility <= 0.8:
        return 3
    elif visibility <= 1:
        return 4
    else:
        return 5


def process_rosbag(bag_file, output_dir):
    bridge = CvBridge() 
    bag = rosbag.Bag(bag_file, 'r')
    bag_name = os.path.basename(bag_file)
    scene_time = os.path.splitext(bag_name)[0]
    scene_name = str(scene_time).split('-')[1] + str(scene_time).split('-')[2] + '-' + str(scene_time).split('-')[3] + str(scene_time).split('-')[4]

    sweep_lidar_folder = os.path.join(output_dir, 'sweeps/LIDAR_TOP') 
    os.makedirs(sweep_lidar_folder, exist_ok=True)

    sweep_camera_folder = os.path.join(output_dir, 'sweeps/CAM_FRONT') 
    os.makedirs(sweep_camera_folder, exist_ok=True)

    sample_lidar_folder = os.path.join(output_dir, 'samples/LIDAR_TOP') 
    os.makedirs(sample_lidar_folder, exist_ok=True)
    
    sample_camera_folder = os.path.join(output_dir, 'samples/CAM_FRONT') 
    os.makedirs(sample_camera_folder, exist_ok=True)

    scenes = []
    samples = []
    sample_datas = []
    sample_annotations = []
    ego_poses = []
    calibrated_sensors = []
    instance_list = []
    sensors = []
    logs = []
    vis_out = []

    save_pcd = True
    save_img = True
    print(f'Reading in a OPEN situation')
    ego_init = np.array([0.0, 0.0, 0.3], dtype=np.float32)

    anno_new_file = '/home/cx/dataset/isaac_sim/info.json'
    anno_test_file = '/home/cx/dataset/isaac_sim/annotations_test.json'

    obstacle_world_coords_dict =[]
    obstacle_world_coords_dict = fill_obs_list(anno_new_file)

    calibrated_sensor_token_camera = "sensor_token_camera_open"
    calibrated_sensor_token_lidar = "sensor_token_lidar_open"

# ===================================== Scene & Log =====================================

    scene_token = f"scene_token_{scene_name}"
    log_token = f"log_token_{scene_name}"
    vehicle = 'Four-Feet Robot'
    logfile = f'{scene_name}_log'
    '''场景是从日志中提取的 20 秒长的连续帧序列。多个场景可以来自同一个日志。请注意，对象身份（实例标记）不会跨场景保留。
        scene {
            "token":                   <str> -- Unique record identifier.
            "name":                    <str> -- Short string identifier.
            "description":             <str> -- Longer description of the scene.
            "log_token":               <str> -- Foreign key. Points to log from where the data was extracted.
            "nbr_samples":             <int> -- Number of samples in this scene.
            "first_sample_token":      <str> -- Foreign key. Points to the first sample in scene.
            "last_sample_token":       <str> -- Foreign key. Points to the last sample in scene.
        }
    '''
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

    '''有关从中提取数据的日志的信息。log表包含从中提取数据的日志信息。日志记录对应于我们的自我车辆沿着预定义路线的一次旅行。让我们检查日志的数量和日志的元数据。
        log {
            "token":                   <str> -- Unique record identifier.
            "logfile":                 <str> -- Log file name.
            "vehicle":                 <str> -- Vehicle name.
            "date_captured":           <str> -- Date (YYYY-MM-DD).
            "location":                <str> -- Area where log was captured, e.g. singapore-onenorth.
        }
    '''
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

    cam_translation = np.array([0.15, 0, 0.4])
    lidar_translation = np.array([0.045, 0, 0.18])

    # Camera Intrinsic
    K = np.array([[958.8, 0, 957.8], [0, 956.7, 589.5], [0, 0, 1]])

    # >>> sensor.json >>>
    '''特定的传感器类型。
        sensor {
            "token":                   <str> -- Unique record identifier.
            "channel":                 <str> -- Sensor channel name.
            "modality":                <str> {camera, lidar, radar} -- Sensor modality. Supports category(ies) in brackets.
        }
    '''
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

    r_Lidar2Ego = np.array([[ 0,  1,  0],   # X_l = -Y_e
                            [-1,  0,  0],   # Y_l =  X_e
                            [ 0,  0,  1]])  # Z_l =  Z_e

    quat_Lidar2Ego = R.from_matrix(r_Lidar2Ego).as_quat()

    r_Cam2Ego = np.array([  [ 0,  0,  1],   # X_c = -Y_e
                            [-1,  0,  0],   # Y_c = -Z_e
                            [ 0, -1,  0]])  # Z_c =  X_e
    
    quat_Cam2Ego = R.from_matrix(r_Cam2Ego).as_quat()

    # >>> calibrated_sensors.json >>>
    '''定义在特定车辆上校准的特定传感器（激光雷达/雷达/摄像机）。所有外部参数都是相对于车身坐标给出的。所有相机图像均未失真且经过校正。
        calibrated_sensor {
            "token":                   <str> -- Unique record identifier.
            "sensor_token":            <str> -- Foreign key pointing to the sensor type.
            "translation":             <float> [3] -- Coordinate system origin in meters: x, y, z.
            "rotation":                <float> [4] -- Coordinate system orientation as quaternion: w, x, y, z.
            "camera_intrinsic":        <float> [3, 3] -- Intrinsic camera calibration. Empty for sensors that are not cameras.
        }
    '''
    calibrated_sensor_camera = {
        'token': calibrated_sensor_token_camera,
        'sensor_token': sensor_token_camera,
        'translation': cam_translation.tolist(), 
        'rotation': [quat_Cam2Ego[3], quat_Cam2Ego[0], quat_Cam2Ego[1], quat_Cam2Ego[2]], 
        'camera_intrinsic': [[958.8, 0, 957.8], [0, 956.7, 589.5], [0, 0, 1]],
    }
    calibrated_sensor_lidar = {
        'token': calibrated_sensor_token_lidar,
        'sensor_token': sensor_token_lidar,
        'translation': lidar_translation.tolist(),
        'rotation': [quat_Lidar2Ego[3], quat_Lidar2Ego[0], quat_Lidar2Ego[1], quat_Lidar2Ego[2]],
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
        odom_timstamp = odom_cache[odom_index - 1][0] if odom_index > 0 else odom_cache[0][0]
        rgb_msg = rgb_cache[rgb_index - 1][1] if rgb_index > 0 else rgb_cache[0][1]
        point_cloud_data = list(pc2.read_points(pc_msg, field_names=("x", "y", "z"), skip_nans=True))
        pc_data_nus = np.array([(-point[1], point[0], point[2]) for point in point_cloud_data]).T
# ======================================== ego_pose ========================================

        '''ego_pose包含关于自车相对于全局坐标系的位置(translation编码)和方向(rotation编码)的信息
        ego_pose {
            "token":                   <str> -- Unique record identifier.
            "translation":             <float> [3] -- Coordinate system origin in meters: x, y, z. Note that z is always 0.
            "rotation":                <float> [4] -- Coordinate system orientation as quaternion: w, x, y, z.
            "timestamp":               <int> -- Unix time stamp.
        }
        '''
        
        ego_pose_token = f"ego_token_{scene_name}_{idx:06d}"
        odom_orientation = odom_msg.pose.pose.orientation
        quaternion = [odom_orientation.x, odom_orientation.y, odom_orientation.z, odom_orientation.w]
        # rotation = R.from_quat(quaternion).as_matrix()
        # quat_inv = R.from_quat(quaternion).inv().as_quat()
        odom_pos = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z])
        t_Global2Ego = ego_init + odom_pos

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
                t_Global2Ego[0],
                t_Global2Ego[1],
                t_Global2Ego[2]
            ],
        }
        ego_poses.append(ego_pose)
# ======================================== ego_pose ========================================
        
        # >>> Save sweeps pcd >>>
        sweep_bin_file = f"pc_{int(pc_timestamp * 1e6)}.pcd.bin"
        sweep_bin_path = os.path.join(sweep_lidar_folder, sweep_bin_file)
        if save_pcd:    
            save_point_cloud_to_pcd(point_cloud_data, sweep_bin_path)
        # <<< Save sweeps pcd <<<

# ================================================= LIDAR =================================================

# ====================================== lidar_sample_data ======================================
        
        '''传感器数据，例如图像、点云或雷达返回。对于 is_key_frame=True 的sample_data,时间戳应该非常接近它指向的样本。对于非关键帧,sample_data 指向时间上最接近的样本。
            sample_data {
                "token":                   <str> -- Unique record identifier.
                "sample_token":            <str> -- Foreign key. Sample to which this sample_data is associated.
                "ego_pose_token":          <str> -- Foreign key.
                "calibrated_sensor_token": <str> -- Foreign key.
                "filename":                <str> -- Relative path to data-blob on disk.
                "fileformat":              <str> -- Data file format.
                "width":                   <int> -- If the sample data is an image, this is the image width in pixels.
                "height":                  <int> -- If the sample data is an image, this is the image height in pixels.
                "timestamp":               <int> -- Unix time stamp.
                "is_key_frame":            <bool> -- True if sample_data is part of key_frame, else False.
                "next":                    <str> -- Foreign key. Sample data from the same sensor that follows this in time. Empty if end of scene.
                "prev":                    <str> -- Foreign key. Sample data from the same sensor that precedes this in time. Empty if start of scene.
            }
        '''
        sd_token_lidar = f"sd_token_{scene_name}_{(idx):06d}"
        sd_lidar = {
            'token': sd_token_lidar,
            'sample_token': sample_token,
            'ego_pose_token': ego_pose_token,
            'calibrated_sensor_token': calibrated_sensor_token_lidar,
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
            if save_pcd:
                save_point_cloud_to_pcd(point_cloud_data, sample_bin_path)

            '''样本是 2 Hz 的带注释的关键帧。作为单次 LIDAR 扫描的一部分，数据在（大约）相同的时间戳处收集。
                sample {
                    "token":                   <str> -- Unique record identifier.
                    "timestamp":               <int> -- Unix time stamp.
                    "scene_token":             <str> -- Foreign key pointing to the scene.
                    "next":                    <str> -- Foreign key. Sample that follows this in time. Empty if end of scene.
                    "prev":                    <str> -- Foreign key. Sample that precedes this in time. Empty if start of scene.
                }
            '''
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

            
            for obs in obstacle_world_coords_dict:
                pos_obs = obs['pos']
                size_obs = obs['size']
                rot_obs = np.array([1,0,0,0])
                
                box_cam = data_classes.Box(pos_obs, size_obs, Quaternion(rot_obs), name = obs['category'])
                box_cam.translate(-np.array(ego_pose['translation']))
                box_cam.rotate(Quaternion(ego_pose['rotation']).inverse)
                box_cam.translate(-np.array(calibrated_sensor_camera['translation']))
                box_cam.rotate(Quaternion(calibrated_sensor_camera['rotation']).inverse)

                if (geometry_utils.box_in_image(box_cam, K, [rgb_msg.width, rgb_msg.height], 1)):
                    
                    box_lidar = data_classes.Box(pos_obs, size_obs, Quaternion(rot_obs), name = obs['category'])
                    box_lidar.translate(-np.array(ego_pose['translation']))
                    box_lidar.rotate(Quaternion(ego_pose['rotation']).inverse)
                    box_lidar.translate(-np.array(calibrated_sensor_lidar['translation']))
                    box_lidar.rotate(Quaternion(calibrated_sensor_lidar['rotation']).inverse)
                    
                    pcd_in_box = geometry_utils.points_in_box(box_lidar, pc_data_nus)

                    if not np.any(pcd_in_box):
                        continue
                    num_lidar_pts = int(np.sum(pcd_in_box))
                        
                    visibility = calculate_visibility(box_cam, K, (rgb_msg.height, rgb_msg.width))
                    if visibility == 5:
                        vis_out.append(obs)
                        continue        
                    print(f'{pos_obs} can be seem at {odom_pos}\n')
                    print(f"visibility level: v-{visibility}\n")
                    print(f"{num_lidar_pts} lidar points\n")
                    sample_anno_token = f'sa_token_{scene_name}_{obs_count:06d}'
                    prev_token = sample_annotations[obs_count - 1]['token'] if obs_count > 0 else ''  
                    next_token = ''  # 默认为空，后续将在列表末尾添加  
                    obs_count = obs_count + 1
                    instance_id = obs["id"]
                    label = obs['label'] 
                    
                    instance_token = f"instance_token_{scene_name}_{instance_id:06d}"
                    
                    '''定义样本中所见对象位置的边界框。所有位置数据都是相对于全局坐标系给出的。
                        sample_annotation {
                            "token":                   <str> -- Unique record identifier.
                            "sample_token":            <str> -- Foreign key. NOTE: this points to a sample NOT a sample_data since annotations are done on the sample level taking all relevant sample_data into account.
                            "instance_token":          <str> -- Foreign key. Which object instance is this annotating. An instance can have multiple annotations over time.
                            "attribute_tokens":        <str> [n] -- Foreign keys. List of attributes for this annotation. Attributes can change over time, so they belong here, not in the instance table.
                            "visibility_token":        <str> -- Foreign key. Visibility may also change over time. If no visibility is annotated, the token is an empty string.
                            "translation":             <float> [3] -- Bounding box location in meters as center_x, center_y, center_z.
                            "size":                    <float> [3] -- Bounding box size in meters as width, length, height.
                            "rotation":                <float> [4] -- Bounding box orientation as quaternion: w, x, y, z.
                            "num_lidar_pts":           <int> -- Number of lidar points in this box. Points are counted during the lidar sweep identified with this sample.
                            "num_radar_pts":           <int> -- Number of radar points in this box. Points are counted during the radar sweep identified with this sample. This number is summed across all radar sensors without any invalid point filtering.
                            "next":                    <str> -- Foreign key. Sample annotation from the same object instance that follows this in time. Empty if this is the last annotation for this object.
                            "prev":                    <str> -- Foreign key. Sample annotation from the same object instance that precedes this in time. Empty if this is the first annotation for this object.
                        }
                    '''
                    sample_annotation = {
                        "token": sample_anno_token,  
                        "sample_token": sample_token,
                        "instance_token": instance_token,
                        "visibility_token": f"{visibility}",  
                        "attribute_tokens": [
                            f"attribute_token_00000{label}"
                        ],
                        "translation": pos_obs,  
                        "size": size_obs,  
                        "rotation": rot_obs.tolist(),
                        "prev": prev_token,  
                        "next": next_token,
                        "num_lidar_pts": num_lidar_pts,
                        "num_radar_pts": 0  
                    }
                    sample_annotations.append(sample_annotation)
            # 更新每个字典中的next字段  
            for i in range(len(sample_annotations) - 1):  
                for j in range(i + 1, len(sample_annotations) - 2):
                    if sample_annotations[i]['instance_token'] == sample_annotations[j]['instance_token']:
                        sample_annotations[i]['next'] = sample_annotations[j]['token']
                        sample_annotations[j]['prev'] = sample_annotations[i]['token']
                        break

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
                    '''对象实例，例如特定车辆。该表是我们观察到的所有对象实例的枚举。请注意，不会跨场景跟踪实例。
                        instance {
                            "token":                   <str> -- Unique record identifier.
                            "category_token":          <str> -- Foreign key pointing to the object category.
                            "nbr_annotations":         <int> -- Number of annotations of this instance.
                            "first_annotation_token":  <str> -- Foreign key. Points to the first annotation of this instance.
                            "last_annotation_token":   <str> -- Foreign key. Points to the last annotation of this instance.
                        }
                    '''
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

        # >>> Save sweeps jpg >>>
        sweep_cam_file = f"img_{int(img_timestamp * 1e6)}.jpg"
        sweep_cam_path = os.path.join(sweep_camera_folder, sweep_cam_file)
        if save_img:
            save_image_to_jpg(rgb_msg, sweep_cam_path, bridge)
        # <<< Save sweeps jpg <<<

# ====================================== sample_data ======================================

        sd_token_camera = f"sd_token_{scene_name}_{(sd_count):06d}"
        ego_pose_token = f"ego_token_{scene_name}_{rgb_idx:06d}"
        sd_camera = {
            'token': sd_token_camera,
            'sample_token': sample_token,
            'ego_pose_token': ego_pose_token,
            'calibrated_sensor_token': calibrated_sensor_token_camera,
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
            if save_img:
                save_image_to_jpg(rgb_msg, sample_cam_path, bridge)
            # print(f"Saving key-frame data at {int(img_timestamp * 1e6)}\n")

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
        'instance': instance_list,
        'vis_out': vis_out
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
    process_rosbag(args.bag, args.output_dir)
