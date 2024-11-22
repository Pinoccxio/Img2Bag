import argparse
import os
import cv2
import numpy as np

import rosbag
import rospy

from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import sensor_msgs.point_cloud2 as pc2



def parse_args():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='读取RosBag包消息')
    parser.add_argument('--bag', type=str, help='RosBag包路径', required=True)
    return parser.parse_args()


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


def save_odometry(msg, timestamp, save_dir):  
    # 保存里程计数据  
    odom_data = {  
        "position": [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z],  
        "orientation": [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w],  
        "timestamp": timestamp  
    }  
    with open(os.path.join(save_dir, 'odometry.txt'), 'a') as f:  
        f.write(str(odom_data) + '\n')  


def save_pcd(msg, timestamp, save_dir):
    # 这里需要转换msg数据为合适的格式，通常是numpy array  
    pc_data = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 4)  
    np.save(os.path.join(save_dir, f'pointcloud_{timestamp}.npy'), pc_data)  


def save_image(msg, timestamp, save_dir):  
    # 保存图像数据  
    img_data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)  
    cv2.imwrite(os.path.join(save_dir, f'sim_image_{timestamp}.png'), img_data)


# 从RosBag读取数据
def extract_and_transform_bag(bag_file):
    # Image Param
    img_width = 1920
    img_height = 1200

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

    # 相机内参矩阵 K
    K = np.array([[958.8, 0, 957.8], [0, 956.7, 589.5], [0, 0, 1]])

    with rosbag.Bag(bag_file, 'r') as bag:
        img_data = []
        timestamps = []
        obstacle_world_coords_dict = {}  # Obs in World
        obstacle_cam_coords_list = {}  # Obs coords after transform

        # 遍历RosBag中的所有消息
        for topic, msg, t in bag.read_messages(topics=['/odom', '/rgb_data', '/pc_scan']):
            if topic == '/odom':
                position = msg.pose.pose.position
                t_body = np.array([position.x, position.y, position.z])
                T_World2Body[:3, 3] = t_World2Body + t_body

                quaternion = msg.pose.pose.orientation
                r_body = R.from_quat(quaternion).as_matrix()
                T_World2Body[:3, :3] = r_body

                T_World2Cam = np.dot(T_Lidar2Cam, np.dot(T_Body2Lidar, T_World2Body))
                print("Transform Matrix From World to Camera: ", T_World2Cam)

                for obs in obstacle_world_coords_dict:
                    pos_obs = np.dot(T_World2Cam, obstacle_world_coords_dict[obs])
                    if is_in_view(pos_obs, K, img_width, img_height):
                        obs_dict = {obs: pos_obs}
                        obstacle_cam_coords_list.update(obs_dict)

            if topic == '/rgb_data':
                image = np.frombuffer(msg.data, dtype=np.uint8).reshape(img_height, img_width, 3)
                img_data.append(image)
                timestamps.append(t.header.stamp.secs)

            if topic == '/pc_scan':  # 处理点云数据
                point_cloud = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
                for p in point_cloud:
                    pcd_coords = np.array(p)  # 每个点的坐标
                    pcd_cam_coords = np.dot(T_Lidar2Cam, pcd_coords)


if __name__ == "__main__":
    args = parse_args()
    RosBag_file = args.bag

    rospy.init_node('transform_coordinates', anonymous=True)
    
    extract_and_transform_bag(RosBag_file)

    # transformed_coords = extract_and_transform_bag(RosBag_file)

    # for i, coords in enumerate(transformed_coords):
    #     print(f'Object {i} in camera coordinates: {coords}')
