import os

# 定义目标文件夹路径
folder_path = '/data/dwj/ReId_without_Id-main/data/full_data_v2/train'
# 定义输出文件路径
output_file_path = '/data/dwj/ReId_without_Id-main/data/full_data_v2/train_list.txt'

# 获取文件名并保持原顺序
file_names = sorted(os.listdir(folder_path))

# 将格式化的文件名写入文本文件
with open(output_file_path, 'w') as f:
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)  # 获取完整的文件路径
        
        # 检查文件是否为空
        if os.path.getsize(file_path) > 0:  # 只有当文件大小大于0时才记录
            # 提取编号部分
            id_part = file_name.split('_')[0]  # 假设编号在文件名的开头
            # 写入格式化字符串
            f.write(f'train/{file_name} {id_part}\n')

print(f'文件名已成功记录到 {output_file_path}')