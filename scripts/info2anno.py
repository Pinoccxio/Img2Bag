import os, argparse
import json
import numpy as np


def parse_args():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='读取RosBag包消息')
    parser.add_argument('--meteor', type=str, help='meter_info.txt', required=True)
    parser.add_argument('--table', type=str, help='table_info.txt', required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    meteor = args.meteor
    table = args.table

    annotations_normal = []
    annotations_oepn = []
    with open(meteor, 'r') as m_txt:
        m_data = json.load(m_txt)
    with open(table, 'r') as t_txt:
        t_data = json.load(t_txt)

    m_id = 0
    for m in m_data:
        m_anno = {
            "x": float(m[0]),
            "y": float(m[1]),
            "z": float(m[2]),
            "w": float(m[3]),
            "l": float(m[4]),
            "h": float(m[5]),
            "category": "meteor",
            "label": 0,
            "id": m_id
        }
        m_id = m_id + 1
        annotations_normal.append(m_anno)
        annotations_oepn.append(m_anno)
    

    for t in t_data:
        t_anno = {
            "x": t[0],
            "y": t[1],
            "z": t[2],
            "w": t[3],
            "l": t[4],
            "h": t[5],
            "category": "table",
            "label": 1,
            "id": m_id
        }
        m_id = m_id + 1
        annotations_normal.append(t_anno)
    
    t_txt.close()
    m_txt.close()

    metadata = {
        'annotations_normal': annotations_normal,
        'annotations_oepn': annotations_oepn
    }

    for key, value in metadata.items():
        with open(os.path.join('/home/cx/dataset/isaac_sim', f'{key}.json'), 'w') as f:
            json.dump(value, f, indent=4)
        print(f"Saved into {key}.json")



if __name__ == "__main__":
    main()
