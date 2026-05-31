import os
import numpy as np
import argparse

def get_folder_coordinate_range(input_folder):
    """
    从文件夹中所有文件获取x和y的坐标范围
    """
    all_x_coords = []
    all_y_coords = []
    
    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    
    print("正在扫描文件以确定坐标范围...")
    for filename in files[:min(10, len(files))]:
        input_path = os.path.join(input_folder, filename)
        try:
            with open(input_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 4:
                        all_x_coords.append(int(parts[1]))
                        all_y_coords.append(int(parts[2]))
        except:
            continue
    
    if all_x_coords and all_y_coords:
        x_range = (0, max(all_x_coords) + 1)
        y_range = (0, max(all_y_coords) + 1)
        print(f"检测到的坐标范围: x=[{x_range[0]}, {x_range[1]}), y=[{y_range[0]}, {y_range[1]})")
        return x_range, y_range
    else:
        print("无法检测坐标范围,使用默认值: x=[0, 640), y=[0, 480)")
        return (0, 640), (0, 480)

def make_sparse(input_file, output_file, keep_ratio=0.5):
    """
    对事件相机数据进行稀疏化处理(随机删除事件)
    
    参数:
        input_file: 输入文件路径
        output_file: 输出文件路径
        keep_ratio: 保留事件的比例 (0-1之间), 例如0.5表示保留50%的事件
    """
    # 读取原始事件数据,保持原始字符串格式
    original_events = []
    
    with open(input_file, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            parts = line.strip().split()
            if len(parts) == 4:
                original_events.append(line)
    
    if len(original_events) == 0:
        # 如果没有有效事件,创建空文件
        with open(output_file, 'w') as f:
            pass
        return 0, 0
    
    num_original = len(original_events)
    
    # 计算要保留的事件数量
    num_keep = int(num_original * keep_ratio)
    
    if num_keep == 0:
        num_keep = 1  # 至少保留一个事件
    
    # 随机选择要保留的事件索引
    keep_indices = np.random.choice(num_original, num_keep, replace=False)
    keep_indices = sorted(keep_indices)  # 保持时间顺序
    
    # 写入输出文件(只写入保留的事件)
    with open(output_file, 'w') as f:
        for idx in keep_indices:
            f.write(original_events[idx] + '\n')
    
    num_removed = num_original - num_keep
    
    return num_original, num_keep, num_removed

def process_folder(input_folder, output_folder, keep_ratio=0.5):
    """
    处理文件夹中的所有事件数据文件
    """
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍历输入文件夹中的所有文件
    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    
    print(f"\n找到 {len(files)} 个文件")
    print(f"保留比例: {keep_ratio*100}%")
    print(f"删除比例: {(1-keep_ratio)*100}%")
    print(f"⚠️  将随机删除 {(1-keep_ratio)*100}% 的事件以模拟稀疏数据\n")
    
    total_original = 0
    total_keep = 0
    total_removed = 0
    
    for i, filename in enumerate(files, 1):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        print(f"[{i}/{len(files)}] 处理: {filename}", end=" ... ")
        
        try:
            num_original, num_keep, num_removed = make_sparse(
                input_path, output_path, keep_ratio
            )
            total_original += num_original
            total_keep += num_keep
            total_removed += num_removed
            print(f"完成 (原始:{num_original}, 保留:{num_keep}, 删除:{num_removed})")
        except Exception as e:
            print(f"失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"处理完成!")
    print(f"总原始事件数: {total_original}")
    print(f"总保留事件数: {total_keep}")
    print(f"总删除事件数: {total_removed}")
    if total_original > 0:
        print(f"实际保留比例: {total_keep/total_original*100:.2f}%")
        print(f"实际删除比例: {total_removed/total_original*100:.2f}%")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='对事件相机数据进行稀疏化处理')
    parser.add_argument('--input_folder', type=str, required=True, help='输入文件夹路径')
    parser.add_argument('--output_folder', type=str, required=True, help='输出文件夹路径')
    parser.add_argument('--keep_ratio', type=float, default=0.5, 
                        help='保留事件的比例 (默认: 0.5, 即保留50%%, 删除50%%)')
    parser.add_argument('--sparse_ratio', type=float, default=None,
                        help='稀疏比例(删除比例), 例如0.3表示删除30%%. 如果指定此参数,将覆盖keep_ratio')
    
    args = parser.parse_args()
    
    # 如果指定了sparse_ratio,则计算keep_ratio
    if args.sparse_ratio is not None:
        keep_ratio = 1.0 - args.sparse_ratio
        if keep_ratio < 0 or keep_ratio > 1:
            print("错误: sparse_ratio 必须在 0-1 之间")
            exit(1)
    else:
        keep_ratio = args.keep_ratio
        if keep_ratio < 0 or keep_ratio > 1:
            print("错误: keep_ratio 必须在 0-1 之间")
            exit(1)
    
    process_folder(
        args.input_folder, 
        args.output_folder, 
        keep_ratio
    )
