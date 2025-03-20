import numpy as np


def generate_depth_map(lidar_points, R, T, camera_matrix, image_shape):
    height, width = image_shape
    depth_map = np.full((height, width), np.inf)

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    for point in lidar_points:
        P_lidar = np.array(point).reshape(3, 1)
        P_camera = R @ P_lidar + T.reshape(3, 1)
        Z = P_camera[2, 0]
        if Z <= 0:
            continue

        X, Y = P_camera[0, 0], P_camera[1, 0]
        u = fx * (X / Z) + cx
        v = fy * (Y / Z) + cy

        if not (0 <= u < width and 0 <= v < height):
            continue

        i, j = int(round(u)), int(round(v))
        if depth_map[j, i] > Z:
            depth_map[j, i] = Z

    depth_map[np.isinf(depth_map)] = 0  # 可选：将无效值设为0
    return depth_map