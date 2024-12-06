instance_token = "instance_token_001"
# 使用分割函数拆分字符串并获取最后一部分
instance_id = int(instance_token.split('_')[-1])
print(instance_id)  # 输出: 123