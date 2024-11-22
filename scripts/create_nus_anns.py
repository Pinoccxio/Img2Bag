import argparse
import os
import cv2
from cv_bridge import CvBridge
import numpy as np
import hashlib
import string
import json

import rospy
import rosbag
import pcl
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, Image
import sensor_msgs.point_cloud2 as pc2


def parse_args():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='读取RosBag包消息')
    parser.add_argument('--bag', type=str, help='RosBag包路径', required=True)
    parser.add_argument('--save_dir', type=str, help='标注保存文件路径', required=True)
    return parser.parse_args()


def key_frame(token):
    if token > 0:
        return True
    return False


def hash_encode(token):
    md = hashlib.md5(str(token).encode())
    md5_passs = md.hexdigest()
    return md5_passs


def save_pcd(point_cloud_msg, pcd_file):
    pc_data = pc2.read_points(point_cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
    pcl_data = pcl.PointCloud()
    pcl_data.from_list(list(pc_data))
    pcl_data.to_file(pcd_file)


def save_jpg(image_msg, image_file):
    bridge = CvBridge()
    cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
    cv2.imwrite(image_file, cv_image)


def anno_scene(scene_list, scene_token, log_token, nbr_samples, first_sample_token, last_sample_token, scene_name):
    scene = {
        "token": scene_token,
        "log_token": log_token,
        "nbr_samples": nbr_samples,
        "first_sample_token": first_sample_token,
        "last_sample_token": last_sample_token,
        "name": scene_name,
        "description": "Simulation Test"
    }
    scene_list.append(scene)


def anno_sample(sample_list, sample_token, timestamp, prev, next, scene_token):
    sample = {
        "token": sample_token,
        "timestamp": timestamp,
        "prev": prev,
        "next": next,
        "scene_token": scene_token
    }
    sample_list.append(sample)


def anno_ego_pose(ego_pose_list, ego_pose_token, timestamp, rotation, translation):
    ego_pose = {
        "token": ego_pose_token,
        "timestamp": timestamp,
        "rotation": rotation,
        "translation": translation
    }
    ego_pose_list.append(ego_pose)


def anno_sample_pcd_data(sample_data_list, pcd_list, is_key_frame, sample_data_prev, sample_data_next, i):
    timestamp, sensor_token, filename, sample_data_token, sample_token, ego_pose_token = pcd_list[i]
    sample_data = {
        "token": sample_data_token,
        "sample_token": sample_token,
        "ego_pose_token": ego_pose_token,
        "calibrated_sensor_token": sensor_token,
        "timestamp": timestamp,
        "fileformat": "pcd",
        "is_key_frame": is_key_frame,
        "height": 0,
        "width": 0,
        "filename": filename,
        "prev": sample_data_prev,
        "next": sample_data_next
    }
    sample_data_list.append(sample_data)


def anno_sample_img_data(sample_data_list, img_list, is_key_frame, sample_data_prev, sample_data_next, i):
    timestamp, sensor_token, filename, sample_data_token, sample_token, ego_pose_token = img_list[i]
    sample_data = {
        "token": sample_data_token,
        "sample_token": sample_token,
        "ego_pose_token": ego_pose_token,
        "calibrated_sensor_token": sensor_token,
        "timestamp": timestamp,
        "fileformat": "jpg",
        "is_key_frame": is_key_frame,
        "height": 1200,
        "width": 1920,
        "filename": filename,
        "prev": sample_data_prev,
        "next": sample_data_next
    }
    sample_data_list.append(sample_data)




def extract_from_bag(bag_file, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with rosbag.Bag(bag_file, 'r') as bag:
        obstacles_coords = [(1,2), (99, 100), (50,50), (37,46)]
        nbr_samples = len(obstacles_coords)

        scene_name = bag_file
        scene_token = hash_encode(scene_name)

        sample_token_list = []          # hash(timestamp_sample）
        sample_data_token_list = []     # hash(timestamp_ego_pose)

        pcd_list = []                   # Tuple(timestamp, sensor_token, fileformat, filename, sample_data_token, sample_token, ego_pose_token)
        img_list = []                   # Tuple(timestamp, sensor_token, fileformat, filename, sample_data_token, sample_token, ego_pose_token)

        scene_list = []                 # scene{}
        sample_list = []                # sample{}
        sample_data_list = []           # sample_data{}
        ego_pose_list = []              # ego_pose{}

        odom_count = 0
        for topic, msg, t in bag.read_messages(topics=['/odom', '/rgb_data', '/pc_scan']):
            

            # 时间戳 
            timestamp = t.to_sec()
            # token
            sample_token = hash_encode(str(timestamp) + '_sample')
            

            
            sample_data_token = ego_pose_token
            sample_data_token_list.append(sample_data_token)

            if topic == '/odom':
                odom_count = odom_count + 1
                ego_pose_token = hash_encode(str(timestamp)+'_ego_pose')
                # 获取四元数
                rotation = [
                    msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                    msg.pose.pose.orientation.w,
                ]
                # 获取平移量
                translation = [
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    msg.pose.pose.position.z,
                ]
                anno_ego_pose(ego_pose_list, ego_pose_token, timestamp, rotation, translation)

            elif topic == '/rgb_data':
                sensor_name = 'CAM_FRONT'
                sensor_token = hash_encode(sensor_name)
                img_dir = os.path.join(save_dir, 'CAM_FRONT')
                filename = f'{t}_{sensor_name}_{timestamp}.jpg'
                save_jpg(msg, os.path.join(img_dir, filename))
                img_list.append((timestamp, sensor_token, filename, sample_data_token, sample_token, ego_pose_token))
        
            elif topic == '/pc_scan':
                sample_token_list.append(sample_token)
                sensor_name = 'LIDAR_TOP'
                sensor_token = hash_encode(sensor_name)
                pcd_dir = os.path.join(save_dir, 'LIDAR_TOP')
                filename = f'{t}_{sensor_name}_{timestamp}.pcd'
                save_pcd(msg, os.path.join(pcd_dir, filename))
                pcd_list.append((timestamp, sensor_token, filename, sample_data_token, sample_token, ego_pose_token))


        first_sample_token = sample_token_list[0]
        last_sample_token = sample_token_list[-1]

        log_name = scene_name + '.log'
        log_token = hash_encode(log_name)

        # Scene
        anno_scene(scene_list, scene_token, log_token, nbr_samples, first_sample_token, last_sample_token, scene_name)
        
        for i in range(0, odom_count):
            # Beginning of list
            if i == 0:                      
                sample_prev = ''
                sample_next = sample_token_list[1]

                sample_data_prev = ''
                sample_data_next = sample_token_list[1]
            # End of List
            elif i == odom_count - 1:       
                sample_prev = sample_token_list[i-1]
                sample_next = ''

                sample_data_prev = sample_data_token_list[i-1]
                sample_data_next = ''
            # Normal Case
            else:                           
                sample_prev = sample_token_list[i-1]
                sample_next = sample_token_list[i+1]

                sample_data_prev = sample_data_token_list[i-1]
                sample_data_next = sample_data_token_list[i+1]
            
            is_key_frame = key_frame(sample_token_list[i])
            anno_sample(sample_list, sample_token, timestamp, sample_prev, sample_next, scene_token)
            anno_sample_pcd_data(sample_data_list, pcd_list, is_key_frame, sample_data_prev, sample_data_next, i)
            anno_sample_img_data(sample_data_list, img_list, is_key_frame, sample_data_prev, sample_data_next, i)


        # 写入ego_pose.json
        with open(os.path.join(save_dir, 'ego_pose.json'), 'a') as pos_ann:
            print("Saving Ego_pose")
            pos_ann.write(json.dumps(ego_pose_list))
        print("ego_pose.json Saved")

        # 写入sample.json
        with open(os.path.join(save_dir, 'sample.json'), 'a') as spl_ann:
            print("Saving Samples")
            spl_ann.write(json.dumps(sample_list))
        print("sample.json Saved")

        # 写入scene.json
        with open(os.path.join(save_dir, 'scene.json'), 'a') as scn_ann:
            print("Saving Scenes")
            scn_ann.write(json.dumps(scene_list))
        print("scene.json Saved")
        

if __name__ == "__main__":
    args = parse_args()
    RosBag_file = args.bag
    save_dir = args.save_dir

    # rospy.init_node('generate annotation', anonymous=True)
    extract_from_bag(RosBag_file, save_dir)
