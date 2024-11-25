import open3d as o3d
import os

class PointCloudVisualizer:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.files = [f for f in os.listdir(folder_path) if f.endswith('.pcd')]
        self.current_index = 0
        self.pcd = None

    def load_next_pcd(self):
        if self.current_index < len(self.files):
            file_path = os.path.join(self.folder_path, self.files[self.current_index])
            print(f"加载点云文件: {file_path}")
            self.pcd = o3d.io.read_point_cloud(file_path)
            self.current_index += 1

    def visualize(self):
        # 初始加载第一个点云文件
        self.load_next_pcd()

        # 创建一个可视化窗口
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()

        # 设置自定义的键盘事件回调
        vis.register_key_callback(262, self.on_right_arrow_key)  # 右箭头键
        vis.register_key_callback(256, self.on_escape_key)       # ESC 键

        vis.add_geometry(self.pcd)
        vis.run()
        vis.destroy_window()

    def on_right_arrow_key(self, vis):
        """按下右箭头键切换到下一个点云文件"""
        if self.current_index < len(self.files):
            self.load_next_pcd()
            vis.clear_geometries()
            vis.add_geometry(self.pcd)
        return False

    def on_escape_key(self, vis):
        """按下ESC键退出"""
        vis.destroy_window()
        return False



# 指定文件夹路径
folder_path = '/home/cx/dataset/30/sweeps/LIDAR_TOP'  # 替换为实际文件夹路径
# 创建可视化对象并开始可视化
visualizer = PointCloudVisualizer(folder_path)
visualizer.visualize()



# import json

# def convert_to_sample_annotation_format(txt_file_path, sample_token, category_token, ego_pose_token, output_json_path):
#     annotations = []

#     # 假设是固定的旋转信息，如果需要动态生成，可以根据情况修改
#     rotation = [0, 0, 0, 1]  # 四元数格式的旋转信息，这里是单位四元数

#     with open(txt_file_path, 'r') as file:
#         data = json.load(file)  # 读取文件内容

#         for i, obj in enumerate(data):
#             # 假设每个物体都来自相同的样本（sample_token），并且没有前后标注（prev, next）
#             annotation = {
#                 "token": f"annotation_{i:04d}",  # 随机生成唯一 token
#                 "sample_token": sample_token,  # 关联的 sample_token
#                 "translation": obj[:3],  # 前3个值是坐标 [x, y, z]
#                 "size": obj[3:],  # 后3个值是尺寸 [width, length, height]
#                 "rotation": rotation,  # 这里使用单位四元数
#                 "category_token": category_token,  # 类别 token
#                 "visibility_token": "",  # 如果没有 visibility 信息，可以为空
#                 "ego_pose_token": ego_pose_token  # ego vehicle 姿态 token
#             }
#             annotations.append(annotation)
    
#     # 保存到 JSON 文件
#     with open(output_json_path, 'w') as output_file:
#         json.dump(annotations, output_file, indent=4)

#     print(f"Annotations saved to {output_json_path}")

# # Example usage
# txt_file_path = "home/cx/dataset/meteor_info.txt"
# sample_token = "sample_token_example"
# category_token = "category_token_example"
# ego_pose_token = "ego_pose_token_example"
# output_json_path = "sample_annotation.json"

# convert_to_sample_annotation_format(txt_file_path, sample_token, category_token, ego_pose_token, output_json_path)


