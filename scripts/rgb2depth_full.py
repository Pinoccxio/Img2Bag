import os, argparse
import numpy as np
from scipy.ndimage import distance_transform_edt
from bisect import bisect_left
import cv2

r_l2c = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
t_l2c = np.array([0, 0.22, -0.105])
K = np.array([[958.8, 0, 957.8], [0, 956.7, 589.5], [0, 0, 1]])

def lidar_to_depth(
        lidar_points: np.array,  # (N, 3) 激光雷达点云
        R: np.array,  # (3, 3) 旋转矩阵
        T: np.array,  # (3,) 平移向量
        camera_matrix: np.array,  # (3, 3) 相机内参矩阵
        image_shape: tuple,  # (H, W) 图像尺寸
        use_bilinear: bool = True,  # 是否使用双线性插值
        fill_holes: bool = True  # 是否填充空洞
) -> np.array:
    """
    将激光雷达点云转换为深度图
    """
    height, width = image_shape
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    # ===== 1. 坐标变换 =====
    # 转换到相机坐标系 (3, N)
    P_cam = (R @ lidar_points.T) + T.reshape(3, 1)

    # 过滤相机后方的点 (Z <= 0)
    valid = P_cam[2, :] > 0
    X, Y, Z = P_cam[0, valid], P_cam[1, valid], P_cam[2, valid]

    # ===== 2. 投影计算 =====
    u = fx * (X / Z) + cx  # (N,)
    v = fy * (Y / Z) + cy  # (N,)

    # ===== 3. 深度图生成 =====
    depth_map = np.full((height, width), np.inf)

    if use_bilinear:
        # 双线性插值模式：影响四个相邻像素
        i = np.floor(u).astype(int)  # (N,)
        j = np.floor(v).astype(int)  # (N,)

        # 生成四个像素坐标
        offsets = [(0, 0), (1, 0), (0, 1), (1, 1)]
        for dx, dy in offsets:
            px = i + dx
            py = j + dy

            # 边界检查
            valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
            px_val, py_val = px[valid], py[valid]
            Z_val = Z[valid]

            # 更新深度图 (保留最小值)
            np.minimum.at(depth_map, (py_val, px_val), Z_val)
    else:
        # 最近邻模式：直接四舍五入
        i = np.round(u).astype(int)
        j = np.round(v).astype(int)

        # 边界检查
        valid = (i >= 0) & (i < width) & (j >= 0) & (j < height)
        i_val, j_val = i[valid], j[valid]
        Z_val = Z[valid]

        # 更新深度图 (保留最小值)
        np.minimum.at(depth_map, (j_val, i_val), Z_val)

    # ===== 4. 后处理 =====
    depth_map[np.isinf(depth_map)] = 0  # 无效区域置零

    if fill_holes:
        # 基于最近邻的空洞填充
        mask = (depth_map == 0)
        if np.any(mask):
            indices = distance_transform_edt(mask, return_indices=True)
            depth_map[mask] = depth_map[tuple(indices[:, mask])]

    return depth_map.astype(np.float32)


def save_depth_map(
        depth_map: np.array,
        filename: str,
        normalize_range: tuple = (0, 50)
):
    """
    保存深度图到文件
    - filename: 输出文件名（不带扩展名）
    - save_png: 是否保存可视化PNG
    - normalize_range: 深度值归一化范围 (min_depth, max_depth)
    """

    # 深度值归一化到0-255
    min_depth, max_depth = normalize_range
    valid_mask = depth_map > 0
    normalized = np.zeros_like(depth_map)
    normalized[valid_mask] = np.clip(
        (depth_map[valid_mask] - min_depth) / (max_depth - min_depth) * 255,
        0, 255
    ).astype(np.uint8)

    # 应用颜色映射并保存
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    cv2.imwrite(filename, colored)


def main():
    pcd_folder = ''
    depth_folder = ''
    os.makedirs(depth_folder, exist_ok=True)
    for pcd_file in sorted(os.listdir(pcd_folder)):
        pcd_path = os.path.join(pcd_folder, pcd_file)
        points = np.fromfile(pcd_path, dtype=np.float32)
        pcd_timestamp = pcd_file.split('_')[1].split('.')[0]

        depth_map = lidar_to_depth(lidar_points=points, R=r_l2c, T=t_l2c, camera_matrix=K, image_shape=(1200, 1920))
        depth_path = os.path.join(depth_folder, f'depth_{pcd_timestamp}.png')
        save_depth_map(
            depth_map,
            depth_path,
            normalize_range=(0, 20)  # 假设有效深度范围0-20米
        )
