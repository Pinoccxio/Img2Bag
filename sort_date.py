import os
import shutil

# 定义源文件夹和目标文件夹
source_folder = 'path/to/your/source_folder'  # 替换为实际路径
destination_folder = 'path/to/your/destination_folder'  # 替换为实际路径

# 创建目标文件夹，如果它不存在
os.makedirs(destination_folder, exist_ok=True)

# 获取文件夹中的所有文件
files = [f for f in os.listdir(source_folder) if f.endswith('.pcd.bin')]

# 根据日期分组文件
date_groups = {}
for file in files:
    # 提取日期字段
    date_field = file.split('-')[2:5]  # 获取2018-08-28部分
    date_key = '-'.join(date_field)

    # 将文件添加到对应的日期组中
    if date_key not in date_groups:
        date_groups[date_key] = []
    date_groups[date_key].append(file)

# 将每个日期组的文件移动到对应的文件夹中
for date, file_list in date_groups.items():
    # 创建以日期命名的子文件夹
    date_folder = os.path.join(destination_folder, date)
    os.makedirs(date_folder, exist_ok=True)

    # 移动文件
    for file_name in file_list:
        source_file = os.path.join(source_folder, file_name)
        destination_file = os.path.join(date_folder, file_name)
        shutil.move(source_file, destination_file)

print("文件已根据日期字段成功分类！")