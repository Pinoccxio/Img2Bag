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


