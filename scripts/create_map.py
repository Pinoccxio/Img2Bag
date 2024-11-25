# import json
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.patches import Rectangle
# import uuid


# category_to_id = {
#     "meteor": 1,
#     "table": 2
# }


# # 生成 NuScenes 格式的 map.json 文件
# def generate_map_json(obstacle_world_coords_dict, output_file, map_image_file):
#     # 为每个地图生成唯一的 token
#     map_token = "map_token_0001" #str(uuid.uuid4().hex)
    
#     # 生成地图图片
#     generate_map_image(obstacle_world_coords_dict, map_image_file)

#     map_data = {
#         "category": "semantic_prior",
#         "token": map_token,
#         "filename": f"maps/{map_token}.png",
#         "log_tokens": []
#     }
    
#     # 保存为 JSON 文件
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(map_data, f, indent=4)
#     print(f"Map JSON saved to {output_file}")

# # 生成对应的地图图片
# def generate_map_image(obstacles, image_file):
#     fig, ax = plt.subplots(figsize=(10, 10))

#     map_width = 100  # 实际地图宽度
#     map_height = 100  # 实际地图高度
#     image_width = 500  # 图像宽度
#     image_height = 500  # 图像高度

#     pixel_per_meter = image_width / map_width
    
#     # 绘制每个障碍物
#     for obs in obstacles:
#         category_id = category_to_id.get(obs["category"], 0)
        
#         # 创建一个矩形表示障碍物，位置 (x, y)，宽度 w 和长度 l
#         rect = Rectangle((obs["pos"][0]+50, obs["pos"][1]+50), obs["size"][0], obs["size"][1], linewidth=2, edgecolor='r', facecolor='r', alpha=0.5)
#         ax.add_patch(rect)
    
#     ax.set_xlim(0, 500)  # 设置 x 轴范围
#     ax.set_ylim(0, 500)  # 设置 y 轴范围
#     ax.set_title("Map of Obstacles")
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
    
#     # 保存地图图片
#     plt.savefig(image_file)
#     plt.close()
#     print(f"Map image saved to {image_file}")

# def main():
#     # 输出文件路径
#     map_json_file = "/home/cx/dataset/map.json"
#     map_image_file = "/home/cx/dataset/sim_map.png"  
    
#     obstacle_world_coords_dict = []
#     json_file = "/home/cx/dataset/annotations.json"
#     # 假设的障碍物数据
#     with open(json_file, 'r', encoding='utf-8') as annotation_json:
#             print("Reading Annotation.json")
#             content = annotation_json.read()
#             annotations = json.loads(content)
#             for anno in annotations:
#                 pos = np.array([anno["x"], anno["y"], anno["z"]])
#                 cat = anno['category']
#                 size = np.array([anno['w'], anno['l'], anno['h']])
#                 obstacle_world_coords_dict.append({"pos": pos, "category": cat, "size": size})

#     # 将类别转换为数字 ID (这里是简单的示例，你可以根据实际需求修改)


#     # 生成 map.json
#     generate_map_json(obstacle_world_coords_dict, map_json_file, map_image_file)
    
# if __name__ == "__main__":
#     main()


import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import uuid

def generate_unique_token():
    """生成唯一的 token"""
    return str(uuid.uuid4().hex)

def fill_obs_list(obstacle_world_coords_dict, json_file):
    """读取障碍物的 JSON 文件并填充到列表中"""
    with open(json_file, 'r', encoding='utf-8') as annotation_json:
        print("Reading Annotation.json")
        content = annotation_json.read()
        annotations = json.loads(content)
        for anno in annotations:
            pos = np.array([anno["x"], anno["y"], anno["z"]])
            cat = anno['category']
            size = np.array([anno['w'], anno['l'], anno['h']])
            obstacle_world_coords_dict.append({"pos": pos, "cat": cat, "size": size})
    print("Annotation.json Closed")

def generate_map_json(obstacles, output_file, map_image_file, map_width=100, map_height=100, image_width=100, image_height=100):
    """
    生成 NuScenes 格式的 map.json 文件
    并生成对应的地图图片

    参数:
    - obstacles: 障碍物列表，包含每个障碍物的坐标和大小
    - output_file: 输出的 map.json 文件路径
    - map_image_file: 输出的地图图片文件路径
    - map_width: 地图的实际宽度（米）
    - map_height: 地图的实际高度（米）
    - image_width: 地图图片的宽度（像素）
    - image_height: 地图图片的高度（像素）
    """
    # 每米对应的像素数
    pixel_per_meter_x = image_width / map_width
    pixel_per_meter_y = image_height / map_height

    # 为每个地图生成唯一的 token
    map_token = generate_unique_token()

    # 生成地图图片
    generate_map_image(obstacles, map_image_file, pixel_per_meter_x, pixel_per_meter_y, image_width, image_height)

    map_data = {
        "category": "semantic_prior",
        "token": map_token,
        "filename": map_image_file,
        "log_tokens": ["log_1234567890abcdef"],  # 示例 log_token，可根据实际情况修改
    }

    # 保存为 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(map_data, f, indent=4)
    print(f"Map JSON saved to {output_file}")

def generate_map_image(obstacles, image_file, pixel_per_meter_x, pixel_per_meter_y, image_width, image_height):
    """
    生成地图图片，绘制障碍物

    参数:
    - obstacles: 障碍物列表，包含每个障碍物的坐标和大小
    - image_file: 输出的地图图片文件路径
    - pixel_per_meter_x: 每米对应的像素数（x 方向）
    - pixel_per_meter_y: 每米对应的像素数（y 方向）
    - image_width: 图像宽度（像素）
    - image_height: 图像高度（像素）
    """
    # 类别颜色映射（根据实际需求调整）
    category_colors = {
        "meteor": "red",
        "table": "blue"
    }

    fig, ax = plt.subplots(figsize=(10, 10))

    # 绘制每个障碍物
    for obs in obstacles:
        category = obs["cat"]
        color = category_colors.get(category, "grey")  # 默认颜色为灰色

        # 归一化坐标到图像像素坐标
        x_map, y_map = obs["pos"][0] + 50, obs["pos"][1] + 50
        x_pixel = x_map #* pixel_per_meter_x
        y_pixel = y_map #* pixel_per_meter_y

        # 归一化障碍物大小到像素
        w_pixel = obs["size"][0] #* pixel_per_meter_x
        l_pixel = obs["size"][1] #* pixel_per_meter_y

        # 创建一个矩形表示障碍物，位置 (x_pixel, y_pixel)，宽度 w_pixel 和长度 l_pixel
        rect = Rectangle((x_pixel, y_pixel), w_pixel, l_pixel, linewidth=1, edgecolor=color, facecolor=color, alpha=0.5)
        ax.add_patch(rect)

    ax.set_xlim(0, image_width)
    ax.set_ylim(0, image_height)
    ax.set_title("Map of Obstacles")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_aspect('equal')  # 保持比例一致

    # 反转 y 轴，使得 y=0 在底部
    ax.invert_yaxis()

    # 保存地图图片
    plt.savefig(image_file)
    plt.close()
    print(f"Map image saved to {image_file}")

def main():
    # 输入文件路径
    anno_file = '/home/cx/dataset/annotations.json'

    # 输出文件路径
    map_json_file = "/home/cx/dataset/map.json"
    map_image_file = "/home/cx/dataset/map_image.png"

    obstacle_world_coords_dict = []
    fill_obs_list(obstacle_world_coords_dict, anno_file)

    # 生成 map.json 和地图图片
    generate_map_json(obstacle_world_coords_dict, map_json_file, map_image_file)

if __name__ == '__main__':
    main()
