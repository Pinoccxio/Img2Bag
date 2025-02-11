# >>> point cloud concatenate >>>
import rosbag
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import os
import json
from tqdm import tqdm
from bisect import bisect_left
from scipy.spatial.transform import Rotation as R

# 打开 ROS bag 文件
bag = rosbag.Bag('/home/cx/dataset/isaac_sim/2025-01-22-19-30-37.bag')
obstacle_json = '/home/cx/dataset/isaac_sim/info.json'
saved_folder = '/home/cx/dataset/isaac_sim/dataset'

# 保存数据的文件路径
output_file = os.path.join(saved_folder, 'pcd.npy')

# 存储所有点云的列表
all_points = []
# batch_size = 500  # 每批处理500个点云
points_cache = []
odom_cache = []
odom_timestamps = []
lidar_translation = np.array([0.045, 0, 0.18])
ego_init = np.array([0.0, 0.0, 0.3], dtype=np.float32)


print('Reading odom')
odom_count = 0
for idx, (topic, msg, t) in enumerate(tqdm(bag.read_messages(topics=['/odom']))):
    if idx % 5 != 0:
        continue
    odom_timestamp = t.to_sec()
    odom_orientation = msg.pose.pose.orientation
    odom_position = msg.pose.pose.position
    odom_timestamps.append(odom_timestamp)
    odom_cache.append((odom_timestamp, msg))
    odom_count += 1
print(f'{odom_count} odom msg saved')
    

print('Reading pcd')
pc_count = 0
batch_points = []  # 用于存储当前批次的点云数据
# 获取点云消息
for idx, (topic, msg, t) in enumerate(tqdm(bag.read_messages(topics=['/pc_scan']))):
    if idx % 5 != 0:
        continue
    pc_idx = idx // 5
    pc_timestamp = t.to_sec()
    pc_data = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)

    odom_index = bisect_left(odom_timestamps, pc_timestamp)
    odom_msg = odom_cache[odom_index - 1][1] if odom_index > 0 else odom_cache[0][1]
    # odom_msg = odom_cache[pc_idx - 1][1] if pc_idx > 0 else odom_cache[0][1]
    odom_orientation = odom_msg.pose.pose.orientation
    quaternion = [odom_orientation.x, odom_orientation.y, odom_orientation.z, odom_orientation.w]
    odom_rot = R.from_quat(quaternion).as_matrix()

    odom_position = odom_msg.pose.pose.position
    odom_pos = np.array([odom_position.x, odom_position.y, odom_position.z])
    

    for point in pc_data:
        point_ego = point - lidar_translation
        point_global = np.dot(odom_rot ,point_ego) + odom_pos + ego_init
        points_cache.append(point_global)

print('Saving to npy')
np.save(output_file, points_cache)

    # # 每达到批次大小，保存并清空当前批次数据
    # if len(batch_points) >= batch_size:
    #     # 将当前批次的数据追加到文件中
    #     if not os.path.exists(output_file):
    #         np.save(output_file, np.array(batch_points))
    #     else:
    #         existing_data = np.load(output_file, allow_pickle=True)
    #         updated_data = np.vstack((existing_data, np.array(batch_points)))
    #         np.save(output_file, updated_data)
        
    #     # 清空当前批次数据
    #     batch_points = []

# # 保存剩余未处理的数据
# if batch_points:
#     if not os.path.exists(output_file):
#         np.save(output_file, np.array(batch_points))
#     else:
#         existing_data = np.load(output_file, allow_pickle=True)
#         updated_data = np.vstack((existing_data, np.array(batch_points)))
#         np.save(output_file, updated_data)

# 关闭 bag 文件
bag.close()
print('bag closed')

all_points = np.load(output_file, allow_pickle=True)
all_points = np.concatenate(all_points, axis=0)
print('pcd_data loaded')
# <<< point cloud export <<<
# ==========================================================================================================
# >>> remove objects >>> 

# 读取 JSON 文件中的障碍物信息
print('reading json ...')
with open(obstacle_json, 'r') as f:
    obstacles_data = json.load(f)

# 创建一个布尔数组，用来标记哪些点需要保留
valid_points = np.ones(len(all_points), dtype=bool)
print('filting ... ')
# 遍历所有障碍物，过滤点云
for obstacle in tqdm(obstacles_data):
    center = np.array([obstacle['x'], obstacle['y'], obstacle['z']])  # 障碍物中心坐标 [x, y, z]
    size = np.array([obstacle['length'], obstacle['width'], obstacle['height']])  # 检测框的大小 [长, 宽, 高]

    # 获取障碍物的边界
    min_bound = center - size / 2
    max_bound = center + size / 2

    # 通过检查点是否在检测框内来过滤点云
    inside_obstacle = (
        (all_points[:, 0] >= min_bound[0]) & (all_points[:, 0] <= max_bound[0]) &
        (all_points[:, 1] >= min_bound[1]) & (all_points[:, 1] <= max_bound[1]) &
        (all_points[:, 2] >= min_bound[2]) & (all_points[:, 2] <= max_bound[2])
    )

    # 将这些点标记为无效
    valid_points &= ~inside_obstacle

# 保留有效的点
cleaned_points = all_points[valid_points]
valid_points_file = os.path.join(saved_folder, 'no_obstacles.npy')
np.save(valid_points_file, cleaned_points)
print('removed obstacles')

# <<< remove objects <<<
# ==========================================================================================================
# >>> classify type >>> 

# 假设 cleaned_points 是已经去除障碍物的点云数据，cleaned_points[:, 2] 是 Z 轴的高度数据
height_values = cleaned_points[:, 2]

# 计算平均高度和标准差
mean_height = np.mean(height_values)
std_height = np.std(height_values)

# 定义三个高度区间
# 地面区域：在平均值附近（±1个标准差）
ground_range = (mean_height - std_height, mean_height + std_height)

# 山坡区域：比平均值高一个标准差以上
slope_range = (mean_height + std_height, np.max(height_values))

# 深坑区域：比平均值低一个标准差以下
pit_range = (np.min(height_values), mean_height - std_height)

# <<< classify type <<<
# ==========================================================================================================
# >>> semantic labeling >>> 

# 为所有点云添加语义标签
semantic_labels = []
print('setting semantic labels')
for point in tqdm(cleaned_points):
    z_value = point[2]
    
    if ground_range[0] <= z_value <= ground_range[1]:
        semantic_labels.append(0)  # 地面
    elif slope_range[0] <= z_value <= slope_range[1]:
        semantic_labels.append(1)  # 山坡
    elif pit_range[0] <= z_value <= pit_range[1]:
        semantic_labels.append(-1)  # 深坑
    else:
        semantic_labels.append(-2)  # 其它区域
print('set semantic labels done')

# 将标签添加到点云数据
batch_size = 1000000  # 每批处理的数据大小
num_batches = len(cleaned_points) // batch_size + 1  # 计算总共需要多少批次

# npy文件（二进制格式存储）
npy_file = os.path.join(saved_folder, 'cleaned_points_with_semantics.npy')

memmap_array = np.lib.format.open_memmap(npy_file, mode='w+', dtype=np.float32, shape=(len(cleaned_points), 4))

for batch_idx in tqdm(range(num_batches)):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, len(cleaned_points))

    batch_cleaned_points = cleaned_points[start_idx:end_idx]
    batch_semantic_labels = semantic_labels[start_idx:end_idx]
    batch_semantic_labels = np.array(batch_semantic_labels).reshape(-1, 1)

    # 拼接当前批次
    batch_cleaned_points_with_semantics = np.hstack((batch_cleaned_points, batch_semantic_labels))
    memmap_array[start_idx:end_idx] = batch_cleaned_points_with_semantics

del memmap_array
print(f'saved into {npy_file}')

# cleaned_points_with_semantics = np.hstack((cleaned_points, np.array(semantic_labels).reshape(-1, 1)))

# <<< semantic labeling <<<
# ==========================================================================================================
# >>> save >>>

# HDF5格式（适用于大数据集）
# import h5py
# h5_file = os.path.join(saved_folder, 'datasetcleaned_points_with_semantics.h5')
# with h5py.File(h5_file, 'w') as f:
#     f.create_dataset('points', data=cleaned_points_with_semantics)      # 保存
#     # loaded_points = f['points'][:]                                    # 加载出来
# print(f'saved into {h5_file}')

# <<< save <<<