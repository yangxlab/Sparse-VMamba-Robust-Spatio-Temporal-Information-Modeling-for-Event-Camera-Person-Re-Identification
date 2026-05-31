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

def add_impulse_noise(input_file, output_file, noise_ratio=0.05, x_range=None, y_range=None):
    """
    为事件相机数据添加脉冲噪声
    保证原始事件数据完全不变,只插入噪声事件
    
    参数:
        input_file: 输入文件路径
        output_file: 输出文件路径
        noise_ratio: 噪声比例 (0-1之间)
        x_range: x坐标范围 (min, max), 如果为None则自动检测
        y_range: y坐标范围 (min, max), 如果为None则自动检测
    """
    # 读取原始事件数据,保持原始字符串格式
    original_events = []  # 存储原始行和解析后的数据
    
    with open(input_file, 'r') as f:
        for idx, line in enumerate(f):
            line = line.rstrip('\n')  # 去除换行符但保留其他格式
            parts = line.strip().split()
            if len(parts) == 4:
                timestamp = float(parts[0])
                x = int(parts[1])
                y = int(parts[2])
                polarity = int(parts[3])
                # 存储: (时间戳, 原始索引, 原始行字符串, 是否为原始事件, x, y, polarity)
                original_events.append((timestamp, idx, line, True, x, y, polarity))
    
    if len(original_events) == 0:
        # 如果没有有效事件,创建空文件
        with open(output_file, 'w') as f:
            pass
        return 0, 0, 0
    
    num_events = len(original_events)
    
    # 如果没有指定范围,自动检测
    if x_range is None or y_range is None:
        x_coords = [e[4] for e in original_events]
        y_coords = [e[5] for e in original_events]
        x_range = (0, max(x_coords) + 1)
        y_range = (0, max(y_coords) + 1)
    
    # 计算需要添加的噪声事件数量
    num_noise = int(num_events * noise_ratio)
    
    # 生成噪声事件
    noise_events = []
    if num_noise > 0:
        # 获取时间范围
        min_time = original_events[0][0]
        max_time = original_events[-1][0]
        
        # 生成噪声事件
        noise_timestamps = np.random.uniform(min_time, max_time, num_noise)
        noise_x = np.random.randint(x_range[0], x_range[1], num_noise)
        noise_y = np.random.randint(y_range[0], y_range[1], num_noise)
        noise_polarity = np.random.randint(0, 2, num_noise)
        
        for i in range(num_noise):
            # 生成噪声事件的字符串表示
            noise_line = f"{noise_timestamps[i]:.8f} {noise_x[i]} {noise_y[i]} {noise_polarity[i]}"
            # 存储: (时间戳, 大索引值(确保排在原始事件后), 字符串, 是否为原始事件, x, y, polarity)
            # 使用 num_events + i 作为索引,确保噪声事件在相同时间戳时排在原始事件后面
            noise_events.append((noise_timestamps[i], num_events + i, noise_line, False, noise_x[i], noise_y[i], noise_polarity[i]))
    
    # 合并原始事件和噪声事件
    all_events = original_events + noise_events
    
    # 按(时间戳, 原始索引)排序,这样可以保持稳定排序
    # 相同时间戳时,原始事件(索引小)会排在噪声事件(索引大)前面
    all_events.sort(key=lambda x: (x[0], x[1]))
    
    # 写入输出文件
    with open(output_file, 'w') as f:
        for event in all_events:
            f.write(event[2] + '\n')  # 写入原始字符串或生成的噪声字符串
    
    return num_events, num_noise, len(all_events)

def process_folder(input_folder, output_folder, noise_ratio=0.05, x_range=None, y_range=None, auto_detect=True):
    """
    处理文件夹中的所有事件数据文件
    """
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 如果启用自动检测且未指定范围,则检测坐标范围
    if auto_detect and (x_range is None or y_range is None):
        x_range, y_range = get_folder_coordinate_range(input_folder)
    
    # 遍历输入文件夹中的所有文件
    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    
    print(f"\n找到 {len(files)} 个文件")
    print(f"噪声比例: {noise_ratio*100}%")
    print(f"使用坐标范围: x={x_range}, y={y_range}")
    print(f"⚠️  原始事件数据将保持完全不变,只插入噪声事件\n")
    
    total_original = 0
    total_noise = 0
    total_final = 0
    
    for i, filename in enumerate(files, 1):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        print(f"[{i}/{len(files)}] 处理: {filename}", end=" ... ")
        
        try:
            num_original, num_noise, num_final = add_impulse_noise(
                input_path, output_path, noise_ratio, x_range, y_range
            )
            total_original += num_original
            total_noise += num_noise
            total_final += num_final
            print(f"完成 (原始:{num_original}, 噪声:{num_noise}, 总计:{num_final})")
        except Exception as e:
            print(f"失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"处理完成!")
    print(f"总原始事件数: {total_original}")
    print(f"总添加噪声数: {total_noise}")
    print(f"总最终事件数: {total_final}")
    if total_original > 0:
        print(f"实际噪声比例: {total_noise/total_original*100:.2f}%")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='为事件相机数据添加脉冲噪声')
    parser.add_argument('--input_folder', type=str, required=True, help='输入文件夹路径')
    parser.add_argument('--output_folder', type=str, required=True, help='输出文件夹路径')
    parser.add_argument('--noise_ratio', type=float, default=0.05, help='噪声比例 (默认: 0.05, 即5%%)')
    parser.add_argument('--x_max', type=int, default=None, help='x坐标最大值 (不指定则自动检测)')
    parser.add_argument('--y_max', type=int, default=None, help='y坐标最大值 (不指定则自动检测)')
    parser.add_argument('--no_auto_detect', action='store_true', help='禁用自动检测坐标范围')
    
    args = parser.parse_args()
    
    # 设置坐标范围
    x_range = (0, args.x_max) if args.x_max is not None else None
    y_range = (0, args.y_max) if args.y_max is not None else None
    
    process_folder(
        args.input_folder, 
        args.output_folder, 
        args.noise_ratio,
        x_range,
        y_range,
        auto_detect=not args.no_auto_detect
    )
