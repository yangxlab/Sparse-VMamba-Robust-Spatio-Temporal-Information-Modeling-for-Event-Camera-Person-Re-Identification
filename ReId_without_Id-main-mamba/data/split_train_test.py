
from PIL import Image
import numpy as np
import cv2
import pdb
import os
import time
import signal
import argparse
import json
import shutil


def split_data(path_, set_):

    path_ = path_ + "/"
    file_array = [file for file in os.listdir(path_) if file.endswith('.txt')]
    file_array = sorted(file_array)
    
    if set_ == "train":
       split_save_dir = "/data/dwj/ReId_without_Id-main/data/full_data_v2/train/"
       step = 1
    else:   
       split_save_dir = "/data/dwj/ReId_without_Id-main/data/full_data_v2/test/"
       step = 5

    if not os.path.exists(split_save_dir):
        os.makedirs(split_save_dir)

    for k in range(0, len(file_array), step):
        copy_path = path_ + file_array[k]
        save_path = split_save_dir + file_array[k]
        shutil.copyfile(copy_path, save_path)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='out_dir_v2/Event_ReId_v2/')
    params = parser.parse_args()

    for i in range(60):
         # 跳过 ID 为 43 的文件夹
         if i == 42:  # 因为 i 从 0 开始，所以 ID 43 对应的索引是 42
               continue
         
         i_d = "{0:03}".format(i + 1)
         path = os.path.join(params.data_dir, i_d)  # 使用 os.path.join 来构建路径
         
         # 检查路径是否存在
         if os.path.exists(path):
               sub_path = os.listdir(path)

               for cam in sub_path:
                  complete_path = os.path.join(path, cam)
                  id_ = i + 1

                  # 根据 ID 确定数据集类型
                  if id_ not in [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 36, 40, 48, 52]:
                     set_ = "train"
                  else:
                     set_ = "test"

                  split_data(complete_path, set_)
         

if __name__ == "__main__":
    main()
