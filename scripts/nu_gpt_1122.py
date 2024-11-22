import os
import json
import shutil
import numpy as np
import rosbag
import rospy
from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
from pyquaternion import Quaternion
from tqdm import tqdm
from bisect import bisect_left
import argparse


def parse_args():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='读取RosBag包消息')
    parser.add_argument('--bag', type=str, help='RosBag包路径', required=True)
    parser.add_argument('--output_dir', type=str, help='标注保存文件路径', required=True)
    return parser.parse_args()


def save_point_cloud_to_pcd(point_cloud_msg, pcd_file):
    import sensor_msgs.point_cloud2 as pc2
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


def process_rosbag(bag_file, output_dir):
    bridge = CvBridge()
    bag = rosbag.Bag(bag_file, 'r')
    bag_name = os.path.basename(bag_file)
    scene_name = os.path.splitext(bag_name)[0]

    # 创建必要的目录
    os.makedirs(os.path.join(output_dir, 'samples/LIDAR_TOP'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'samples/CAM_FRONT'), exist_ok=True)

    # 初始化元数据列表
    scenes = []
    samples = []
    sample_datas = []
    ego_poses = []
    calibrated_sensors = []
    sensors = []
    logs = []
    maps = []
    sample_annotations = []
    instances = []
    categories = ["meteor", "table"]
    attributes = []
    visibilities = []

    # 假设只有一个场景和一个日志
    scene_token = f"scene_token_{scene_name}"
    log_token = f"log_token_{scene_name}"

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
    
    vehicle = 'Four-Feet Robot'
    logfile = f'{scene_name}_log'
    log = {
        'token': log_token,
        'vehicle': vehicle,
        'date_captured': 'Go See Bag Info',
        'location': 'Issac Sim',
        'logfile': logfile,
        'vehicle_name': 'Sim Car',
        'scene_token': scene_token,
    }
    logs.append(log)

    # 定义传感器信息
    sensor_token_camera = "sensor_token_camera"
    sensor_token_lidar = "sensor_token_lidar"

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

    # 假设传感器的校准参数已知，这里使用默认值
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
    }
    calibrated_sensors.extend([calibrated_sensor_camera, calibrated_sensor_lidar])

    # 初始化索引和缓存
    odom_cache = []
    rgb_cache = []
    odom_timestamps = []
    rgb_timestamps = []

    # 首先，将 /odom 和 /rgb_data 的消息索引建立起来
    print("Indexing /odom and /rgb_data messages...")
    for topic, msg, t in tqdm(bag.read_messages(topics=['/odom', '/rgb_data'])):
        timestamp = t.to_sec()
        if topic == '/odom':
            odom_cache.append((timestamp, msg))
            odom_timestamps.append(timestamp)
        elif topic == '/rgb_data':
            rgb_cache.append((timestamp, msg))
            rgb_timestamps.append(timestamp)

    # 按时间排序
    odom_timestamps, odom_cache = zip(*sorted(zip(odom_timestamps, odom_cache)))
    rgb_timestamps, rgb_cache = zip(*sorted(zip(rgb_timestamps, rgb_cache)))

    # 处理 /pc_scan 消息，逐条处理，避免占用大量内存
    print("Processing /pc_scan messages...")
    previous_sample_token = ''
    first_sample_token = ''
    last_sample_token = ''

    odom_timestamps = list(odom_timestamps)
    rgb_timestamps = list(rgb_timestamps)

    for idx, (topic, pc_msg, t) in enumerate(tqdm(bag.read_messages(topics=['/pc_scan']))):
        pc_timestamp = t.to_sec()

        sample_token = f"sample_token_{idx:06d}"
        if idx == 0:
            first_sample_token = sample_token

        # 查找与当前时间最接近的 odom 消息
        odom_index = bisect_left(odom_timestamps, pc_timestamp)
        if odom_index == 0:
            odom_msg = odom_cache[0][1]
        elif odom_index == len(odom_timestamps):
            odom_msg = odom_cache[-1][1]
        else:
            before = odom_cache[odom_index - 1]
            after = odom_cache[odom_index]
            if abs(before[0] - pc_timestamp) < abs(after[0] - pc_timestamp):
                odom_msg = before[1]
            else:
                odom_msg = after[1]

        # 创建 ego_pose
        ego_pose_token = f"ego_pose_token_{idx:06d}"
        odom_orientation = odom_msg.pose.pose.orientation
        odom_position = odom_msg.pose.pose.position
        ego_pose = {
            'token': ego_pose_token,
            'timestamp': int(pc_timestamp * 1e6),
            'rotation': [
                odom_orientation.w,
                odom_orientation.x,
                odom_orientation.y,
                odom_orientation.z,
            ],
            'translation': [
                odom_position.x,
                odom_position.y,
                odom_position.z,
            ],
        }
        ego_poses.append(ego_pose)

        # 保存点云数据
        pc_filename = f"samples/LIDAR_TOP/pc_{idx:06d}.pcd"
        pc_filepath = os.path.join(output_dir, pc_filename)
        save_point_cloud_to_pcd(pc_msg, pc_filepath)

        # 查找与当前时间最接近的 rgb 消息
        rgb_index = bisect_left(rgb_timestamps, pc_timestamp)
        if rgb_index == 0:
            rgb_msg = rgb_cache[0][1]
        elif rgb_index == len(rgb_timestamps):
            rgb_msg = rgb_cache[-1][1]
        else:
            before = rgb_cache[rgb_index - 1]
            after = rgb_cache[rgb_index]
            if abs(before[0] - pc_timestamp) < abs(after[0] - pc_timestamp):
                rgb_msg = before[1]
            else:
                rgb_msg = after[1]

        # 保存图像数据
        img_filename = f"samples/CAM_FRONT/img_{idx:06d}.jpg"
        img_filepath = os.path.join(output_dir, img_filename)
        save_image_to_jpg(rgb_msg, img_filepath, bridge)

        # 创建样本数据（点云）
        sd_token_lidar = f"sample_data_lidar_token_{idx:06d}"
        sd_lidar = {
            'token': sd_token_lidar,
            'sample_token': sample_token,
            'ego_pose_token': ego_pose_token,
            'calibrated_sensor_token': sensor_token_lidar,
            'timestamp': int(pc_timestamp * 1e6),
            'fileformat': 'pcd',
            'is_key_frame': True,
            'height': 0,
            'width': 0,
            'filename': pc_filename,
            'prev': '',  # 之后更新
            'next': '',  # 之后更新
        }
        sample_datas.append(sd_lidar)

        # 创建样本数据（图像）
        sd_token_camera = f"sample_data_camera_token_{idx:06d}"
        sd_camera = {
            'token': sd_token_camera,
            'sample_token': sample_token,
            'ego_pose_token': ego_pose_token,
            'calibrated_sensor_token': sensor_token_camera,
            'timestamp': int(pc_timestamp * 1e6),
            'fileformat': 'jpg',
            'is_key_frame': True,
            'height': rgb_msg.height,
            'width': rgb_msg.width,
            'filename': img_filename,
            'prev': '',  # 之后更新
            'next': '',  # 之后更新
        }
        sample_datas.append(sd_camera)

        # 创建样本
        sample = {
            'token': sample_token,
            'timestamp': int(pc_timestamp * 1e6),
            'prev': previous_sample_token,
            'next': '',  # 之后更新
            'scene_token': scene_token,
        }
        samples.append(sample)

        # 更新前一帧的 next
        if previous_sample_token != '':
            samples[-2]['next'] = sample_token
            sample_datas[-4]['next'] = sd_token_lidar  # 更新前一帧的 sample_data next
            sample_datas[-3]['next'] = sd_token_camera

            sd_lidar['prev'] = sample_datas[-4]['token']
            sd_camera['prev'] = sample_datas[-3]['token']
        else:
            sd_lidar['prev'] = ''
            sd_camera['prev'] = ''

        previous_sample_token = sample_token
        last_sample_token = sample_token

    # 更新场景的 first_sample_token 和 last_sample_token
    scenes[0]['first_sample_token'] = first_sample_token
    scenes[0]['last_sample_token'] = last_sample_token
    scenes[0]['nbr_samples'] = len(samples)

    # 将元数据保存为 JSON 文件
    metadata = {
        'scene': scenes,
        'sample': samples,
        'sample_data': sample_datas,
        'ego_pose': ego_poses,
        'calibrated_sensor': calibrated_sensors,
        'sensor': sensors,
        'log': logs,
        # 'map': maps,
        # 'sample_annotation': sample_annotations,
        # 'instance': instances,
        'category': categories,
        # 'attribute': attributes,
        # 'visibility': visibilities,
    }

    for key, value in metadata.items():
        with open(os.path.join(output_dir, f'{key}.json'), 'w') as f:
            json.dump(value, f, indent=4)

    bag.close()


def main():
    args = parse_args()
    bag_file = args.bag
    output_dir = args.output_dir

    process_rosbag(bag_file, output_dir)

if __name__ == '__main__':
    main()
