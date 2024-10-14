import pandas as pd
import os
import json
from concurrent.futures import ThreadPoolExecutor

# 指定要忽略的特征
ignore_columns = ['segment', 'timestep']

def calculate_file_min_max(file_path):
    """
    计算单个文件中每个特征的最小值和最大值。
    """
    try:
        print(f"正在处理文件: {file_path}")
        data = pd.read_csv(file_path)

        # 提取需要计算最值的特征（排除指定的忽略特征）
        features = [col for col in data.columns if col not in ignore_columns]

        # 初始化文件的最小值和最大值字典
        file_min_max = {feature: {'min': data[feature].min(), 'max': data[feature].max()} for feature in features}
        return os.path.basename(file_path), file_min_max
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None, None

def calculate_global_min_max(base_dir, all_files_output_path, global_output_path, max_workers=4):
    """
    计算所有文件的最小值和最大值，并保存到指定文件。
    """
    # 获取所有CSV文件的路径
    file_paths = [os.path.join(base_dir, file) for file in os.listdir(base_dir) if file.endswith('.csv')]

    # 使用ThreadPoolExecutor并行处理每个文件
    all_file_min_max = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(calculate_file_min_max, file_paths))

    # 合并每个文件的最值
    for file_name, file_min_max in results:
        if file_name and file_min_max:
            all_file_min_max[file_name] = file_min_max

    # 保存所有文件的最值到 JSON 文件
    with open(all_files_output_path, 'w') as f:
        json.dump(all_file_min_max, f, indent=4)
    print(f"所有文件的最小值和最大值已保存到: {all_files_output_path}")

    # 初始化全局最小值和最大值的字典
    global_min_max = {}

    # 计算全局最值
    for file_min_max in all_file_min_max.values():
        for feature, min_max in file_min_max.items():
            if feature not in global_min_max:
                global_min_max[feature] = {'min': float('inf'), 'max': float('-inf')}
            global_min_max[feature]['min'] = min(global_min_max[feature]['min'], min_max['min'])
            global_min_max[feature]['max'] = max(global_min_max[feature]['max'], min_max['max'])

    # 保存全局最小值和最大值到 JSON 文件
    with open(global_output_path, 'w') as f:
        json.dump(global_min_max, f, indent=4)
    print(f"全局最小值和最大值已保存到: {global_output_path}")

# 主程序
base_dir = 'E:\\climbing-aircraft-dataset\\lat_lon_diff_data'  # 处理后的数据目录
all_files_output_path = 'E:\\climbing-aircraft-dataset\\all_files_min_max.json'  # 所有文件最值保存路径
global_output_path = 'E:\\climbing-aircraft-dataset\\global_min_max.json'  # 全局最值保存路径
calculate_global_min_max(base_dir, all_files_output_path, global_output_path, max_workers=4)
