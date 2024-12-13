import os, argparse
import json
import numpy as np
import shutil


ego_pose_dict = []
instance_dict = []
log_dict = []
sample_annotation_dict = []
sample_data_dict = []
sample_dict = []
scene_dict = []


def parse_args():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='整理nuscenes格式')
    parser.add_argument('--root', type=str, help='root_dir', required=True)
    parser.add_argument('--output', type=str, help='output', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.root
    output = args.output
    os.makedirs(output, exist_ok=True)
    json_list = ['ego_pose.json', 'instance.json', 'log.json', 'sample_annotation.json', 'sample_data.json', 'sample.json', 'scene.json']
    for scene in sorted(os.listdir(root)):
        scene_path = os.path.join(root, scene)
        for folder in sorted(os.listdir(scene_path)):
            folder_path = os.path.join(scene_path, folder)
            if folder == 'samples' or folder == 'sweeps':
                for sensor_folder in sorted(os.listdir(folder_path)):
                    sensor_folder_path = os.path.join(folder_path, sensor_folder)
                    for file in sorted(os.listdir(sensor_folder_path)):
                        src = os.path.join(sensor_folder_path, file)
                        dst_folder = os.path.join(output, folder, sensor_folder)
                        os.makedirs(dst_folder, exist_ok=True)
                        dst = os.path.join(dst_folder, file)
                        shutil.move(src,dst)
                        print(f'{scene}-{folder}: copying {src} to {dst}')
            if folder == 'v1.0-mini':
                for json_file in sorted(os.listdir(folder_path)):
                    if json_file in json_list:  # instance/.../sample.json
                        print(f'reading {json_file}')
                        json_path = os.path.join(folder_path, json_file)
                        with open(json_path, 'r') as j_file:
                            j_data = json.load(j_file)
                            for dict in j_data:
                                if json_file == 'ego_pose.json':
                                    ego_pose_dict.append(dict)
                                elif json_file == 'instance.json':
                                    instance_dict.append(dict)
                                elif json_file == 'log.json':
                                    log_dict.append(dict)
                                elif json_file == 'sample_annotation.json':
                                    sample_annotation_dict.append(dict)
                                elif json_file == 'sample_data.json':
                                    sample_data_dict.append(dict)
                                elif json_file == 'sample.json':
                                    sample_dict.append(dict)
                                elif json_file == 'scene.json':
                                    scene_dict.append(dict)
    
    metadata = {
        'scene': scene_dict,
        'sample': sample_dict,
        'sample_data': sample_data_dict,
        'sample_annotation': sample_annotation_dict,
        'ego_pose': ego_pose_dict,
        'log': log_dict,
        'instance': instance_dict
    }
    print("Ready to write json files")
    v_path = os.path.join(output, 'v1.0-mini')
    os.makedirs(v_path, exist_ok=True)
    for key, value in metadata.items():
        with open(os.path.join(v_path, f'{key}.json'), 'w') as f:
            json.dump(value, f, indent=4)
        print(f"Saved into {key}.json")


if __name__ == "__main__":
    main()