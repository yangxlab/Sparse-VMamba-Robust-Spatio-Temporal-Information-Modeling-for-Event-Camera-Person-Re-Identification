import os
import random
from collections import defaultdict

# 定义目标文件夹路径
folder_path = '/data/dwj/ReId_without_Id-main/data/full_data_v2/test'
# 定义输出文件路径
query_file_path = '/data/dwj/ReId_without_Id-main/data/full_data_v2/query_list.txt'
gallery_file_path = '/data/dwj/ReId_without_Id-main/data/full_data_v2/gallery_list.txt'

# 创建一个字典以存储每个ID和相机的文件
id_camera_map = defaultdict(list)

# 遍历文件夹，按ID和相机分组文件
for file_name in sorted(os.listdir(folder_path)):
    if file_name.endswith('.txt'):
        file_path = os.path.join(folder_path, file_name)  # 获取完整的文件路径
        
        # 检查文件是否为空
        if os.path.getsize(file_path) > 0:  # 只有当文件大小大于0时才处理
            parts = file_name.split('_')
            if len(parts) == 3:
                id_part = parts[0]  # ID
                camera_part = parts[1]  # Camera
                id_camera_map[(id_part, camera_part)].append(file_name)

# 初始化查询和图库列表
query_list = []
gallery_list = []

# 从每个ID每个相机中随机选择一个编号
for (id_part, camera_part), files in id_camera_map.items():
    if files:
        # 随机选择一个文件作为查询
        query_file = random.choice(files)
        query_list.append(f'test/{query_file} {id_part}')
        
        # 将其他文件添加到图库列表
        for file_name in files:
            if file_name != query_file:
                gallery_list.append(f'test/{file_name} {id_part}')

# 将查询列表写入文件
with open(query_file_path, 'w') as q_file:
    for entry in query_list:
        q_file.write(entry + '\n')

# 将图库列表写入文件
with open(gallery_file_path, 'w') as g_file:
    for entry in gallery_list:
        g_file.write(entry + '\n')

print(f'查询列表已成功记录到 {query_file_path}')
print(f'图库列表已成功记录到 {gallery_file_path}')