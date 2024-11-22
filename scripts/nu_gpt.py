# pip install rospy rosbag pyquaternion nuscenes-devkit
# pip install opencv-python
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

import argparse
import hashlib


def parse_args():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='读取RosBag包消息')
    parser.add_argument('--bag', type=str, help='RosBag包路径', required=True)
    parser.add_argument('--output_dir', type=str, help='标注保存文件路径', required=True)
    return parser.parse_args()


def hash_encode(token):
    md = hashlib.md5(str(token).encode())
    md5_passs = md.hexdigest()
    return md5_passs


def find_closest_msg(timestamp, msgs):
    # 简单地遍历找到时间上最近的消息
    min_diff = float('inf')
    closest_msg = None
    for msg_timestamp, msg in msgs:
        diff = abs(msg_timestamp - timestamp)
        if diff < min_diff:
            min_diff = diff
            closest_msg = msg
            closest_timestamp = msg_timestamp
    return closest_msg, closest_timestamp


def save_point_cloud_to_pcd(point_cloud_msg, pcd_file):
    # 将 ROS PointCloud2 消息转换为 PCD 文件
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


# 提取RosBag数据
def extract_rosbag_data(bag_file, bag_name, output_dir):
    bridge = CvBridge()
    bag = rosbag.Bag(bag_file, 'r')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f'{output_dir} create successfully!\n')

    odom_msgs = []
    pc_scan_msgs = []
    rgb_data_msgs = []

    print(f'Reading {bag_name}\n')
    for topic, msg, t in tqdm(bag.read_messages()):
        timestamp = t.to_sec()

        if topic == '/odom':
            odom_msgs.append((timestamp, msg))
        elif topic == '/pc_scan':
            pc_scan_msgs.append((timestamp, msg))
        elif topic == '/rgb_data':
            rgb_data_msgs.append((timestamp, msg))

    bag.close()
    print(f'Successfully extracted')
    print(f'Numbers of odom_msg: {len(odom_msgs)}')
    print(f'Numbers of pc_msg: {len(pc_scan_msgs)}')
    print(f'Numbers of rgb_msg: {len(rgb_data_msgs)}')
    return odom_msgs, pc_scan_msgs, rgb_data_msgs


# 生成nuscenes元数据表
def create_nuscenes_metadata(output_dir, scene_name, odom_msgs, pc_scan_msgs, rgb_data_msgs):
    # 初始化元数据列表
    scenes = []
    samples = []
    sample_datas = []
    ego_poses = []
    calibrated_sensors = []
    sensors = []
    logs = []
    # maps = []
    # sample_annotations = []  # 如果有标注信息
    # instances = []  # 如果有标注信息
    categories = ["meteor", "table"]  # 如果有标注信息
    # attributes = []  # 如果有标注信息
    # visibility = []  # 如果有标注信息

    # 创建必要的目录
    os.makedirs(os.path.join(output_dir, 'samples/LIDAR_TOP'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'samples/CAM_FRONT'), exist_ok=True)

    # Scene & Log
    scene_token = f'scene_token_{scene_name}'
    log_token = f'log_token_{scene_name}'
    
    scene = {
        'token': scene_token,
        'name': scene_name,
        'description': 'Moon Example',
        'log_token': log_token,
        'nbr_samples': len(pc_scan_msgs),
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
    sensor_token_camera = "sensor_token_CAM_FRONT"
    sensor_token_lidar = "sensor_token_LIDAR_TOP"

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

    # 初始化
    previous_sample_token = ''
    first_sample_token = ''
    last_sample_token = ''

    for idx, (pc_timestamp, pc_msg) in enumerate(tqdm(pc_scan_msgs)):
        sample_token = f"sample_token_{idx:06d}"
        if idx == 0:
            first_sample_token = sample_token

        # 匹配对应的 odom 消息
        odom_msg, odom_timestamp = find_closest_msg(pc_timestamp, odom_msgs)
        ego_pose_token = f"ego_pose_token_{idx:06d}"
        ego_pose = {
            'token': ego_pose_token,
            'timestamp': int(odom_timestamp * 1e6),
            'translation': [
                odom_msg.pose.pose.position.x,
                odom_msg.pose.pose.position.y,
                odom_msg.pose.pose.position.z,
            ],
            'rotation': [
                odom_msg.pose.pose.orientation.x,
                odom_msg.pose.pose.orientation.y,
                odom_msg.pose.pose.orientation.z,
                odom_msg.pose.pose.orientation.w,
            ],
        }
        ego_poses.append(ego_pose)

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
            'filename': f"samples/LIDAR_TOP/LIDAR_TOP_{pc_timestamp}.pcd",
            'prev': '',  # 之后更新
            'next': '',  # 之后更新
            'sensor_modality': 'lidar',
            'channel': 'LIDAR_TOP',
        }
        sample_datas.append(sd_lidar)

        # 创建样本数据（图像）
        rgb_msg, rgb_timestamp = find_closest_msg(pc_timestamp, rgb_data_msgs)
        sd_token_camera = f"sample_data_camera_token_{idx:06d}"
        sd_camera = {
            'token': sd_token_camera,
            'sample_token': sample_token,
            'ego_pose_token': ego_pose_token,
            'calibrated_sensor_token': sensor_token_camera,
            'timestamp': int(pc_timestamp * 1e6),
            'fileformat': 'jpg',
            'is_key_frame': True,
            'height': 1200,
            'width': 1920,
            'filename': f"samples/CAM_FRONT/CAM_FRONT_{rgb_timestamp}.jpg",
            'prev': '',  # 之后更新
            'next': '',  # 之后更新
            'sensor_modality': 'camera',
            'channel': 'CAM_FRONT',
        }
        sample_datas.append(sd_camera)

        # 创建样本
        sample = {
            'token': sample_token,
            'timestamp': int(pc_timestamp * 1e6),
            'scene_token': scene_token,
            'data': {
                'LIDAR_TOP': sd_token_lidar,
                'CAM_FRONT': sd_token_camera,
            },
            'prev': previous_sample_token,
            'next': '',  # 之后更新
            'anns': [],  # 如果有标注信息，需要填写
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

    # 如果有地图信息，可以添加到 maps 列表
    # 如果有标注信息，需要生成 instance.json、sample_annotation.json、category.json、attribute.json、visibility.json 等

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
        # 如果有标注信息，添加以下内容
        # 'sample_annotation': sample_annotations,          # TODO
        # 'instance': instances,                            # TODO
        'category': categories,
        # 'attribute': attributes,
        # 'visibility': visibility,
    }

    for key, value in metadata.items():
        with open(os.path.join(output_dir, f'{key}.json'), 'w') as f:
            json.dump(value, f)


# 保存传感器数据
def save_sensor_data(output_dir, pc_scan_msgs, rgb_data_msgs):
    # 创建目录
    lidar_dir = os.path.join(output_dir, 'samples/LIDAR_TOP')
    camera_dir = os.path.join(output_dir, 'samples/CAM_FRONT')
    os.makedirs(lidar_dir, exist_ok=True)
    os.makedirs(camera_dir, exist_ok=True)

    bridge = CvBridge()

    for idx, (pc_timestamp, pc_msg) in enumerate(tqdm(pc_scan_msgs)):
        # 保存点云数据
        pc_filename = os.path.join(lidar_dir, f"{pc_timestamp}_LIDAR_TOP.pcd")
        save_point_cloud_to_pcd(pc_msg, pc_filename)

        # 找到时间上最近的图像
        rgb_msg, rgb_timestamp = find_closest_msg(pc_timestamp, rgb_data_msgs)
        img_filename = os.path.join(camera_dir, f"{rgb_timestamp}_CAM_FRONT.jpg")
        save_image_to_jpg(rgb_msg, img_filename, bridge)


def main():
    args = parse_args()
    bag_file = args.bag
    output_dir = args.output_dir

    bag_name = os.path.basename(bag_file)
    scene_name = os.path.splitext(bag_name)[0]

    # 提取数据
    odom_msgs, pc_scan_msgs, rgb_data_msgs = extract_rosbag_data(bag_file, bag_name, output_dir)

    # 保存传感器数据
    save_sensor_data(output_dir, pc_scan_msgs, rgb_data_msgs)
    
    # 创建 nuScenes 元数据
    create_nuscenes_metadata(output_dir, scene_name, odom_msgs, pc_scan_msgs, rgb_data_msgs)


if __name__ == '__main__':
    main()
