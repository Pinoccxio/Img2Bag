import json

# 假设我们已经有了 annotation.json 和 sample_annotation.json
# annotation_file = '/home/cx/dataset/annotations.json'
sample_annotation_file = '/home/cx/dataset/isaac_sim/dataset/0930_0910/v1.0-mini/sample_annotation.json'
instance_output_file = '/home/cx/dataset/isaac_sim/dataset/0930_0910/v1.0-mini/instance.json'


def generate_instance_data(sample_annotations):
    # 用于存储每个 instance 的相关数据
    instance_data = {}
    sorted_sample_annotations = sorted(sample_annotations, key=lambda x: x['token'])
    # 遍历所有的 sample_annotation，来创建 instance_data
    for sample_annotation in sorted_sample_annotations:
        instance_token = sample_annotation['instance_token']
        sample_annotation_token = sample_annotation['token']
        cat_id = int(sample_annotation['attribute_tokens'][0].split('_')[-1])
        category_token = f"category_token_{cat_id:06d}"

        if instance_token not in instance_data:
            instance_data[instance_token] = {
                "token": instance_token,
                "category_token": category_token,  # 可根据实际需求设置类别标识符
                "nbr_annotations": 0,
                "first_annotation_token": sample_annotation_token,
                "last_annotation_token": sample_annotation_token
            }

        # 更新 last_annotation_token
        instance_data[instance_token]["last_annotation_token"] = sample_annotation_token
        instance_data[instance_token]["nbr_annotations"] = instance_data[instance_token]["nbr_annotations"] + 1

    # 将 instance_data 转换为列表格式
    instance_list = list(instance_data.values())
    return instance_list

def update_sample_annotation_with_instance_tokens():
    with open(sample_annotation_file, 'r') as sa_file:
        sample_annotations = json.load(sa_file)

    # 更新 instance_data
    instance_data = generate_instance_data(sample_annotations)
    
    # 写入更新后的 instance.json
    with open(instance_output_file, 'w') as f:
        json.dump(instance_data, f, indent=4)
    

# 主流程
def process_data():

    # 更新 sample_annotation 中的 instance_token 并生成 instance.json
    update_sample_annotation_with_instance_tokens()

# 运行数据处理
process_data()
