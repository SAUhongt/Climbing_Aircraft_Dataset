import pandas as pd
import os
import json
from concurrent.futures import ThreadPoolExecutor
import logging

# 指定要忽略的特征
ignore_columns = ['segment', 'timestep']

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def normalize_with_global_min_max(data, global_min_max):
    # 对除忽略特征外的所有特征进行归一化
    for feature in data.columns:
        if feature in ignore_columns or feature.endswith('_normalized'):
            continue
        if feature not in global_min_max:
            logging.warning(f"特征 {feature} 不在全局最值中，跳过归一化。")
            continue

        min_val = global_min_max[feature]['min']
        max_val = global_min_max[feature]['max']
        if max_val != min_val:  # 防止除以零
            data[f'{feature}_normalized'] = (data[feature] - min_val) / (max_val - min_val)
            logging.info(f"特征 {feature} 已归一化。")
        else:
            data[f'{feature}_normalized'] = 0  # 如果所有值相同，则归一化为0
            logging.info(f"特征 {feature} 最小值等于最大值，已归一化为 0。")

    return data

def normalize_data_with_global_min_max(file_path, output_path, global_min_max):
    try:
        # 加载数据
        logging.info(f"开始处理文件: {file_path}")
        data = pd.read_csv(file_path)
        logging.info(f"文件 {file_path} 已加载，共 {len(data)} 条记录。")

        # 对数据进行归一化
        normalized_data = normalize_with_global_min_max(data, global_min_max)

        # 保留'segment'和'timestep'以及归一化后的特征
        final_columns = ['segment', 'timestep'] + [col for col in normalized_data.columns if col.endswith('_normalized')]
        normalized_data = normalized_data[final_columns]

        # 保存归一化后的数据
        normalized_data.to_csv(output_path, index=False)
        logging.info(f"已保存归一化数据: {output_path}")
    except Exception as e:
        logging.error(f"处理文件时发生错误: {file_path}, 错误信息: {e}")

def main():
    base_dir = 'E:\\climbing-aircraft-dataset\\lat_lon_diff_data'  # 处理后的数据目录
    output_dir = 'E:\\climbing-aircraft-dataset\\normalized_data'  # 输出归一化后的数据目录
    os.makedirs(output_dir, exist_ok=True)  # 创建输出目录

    # 加载全局最值
    try:
        with open('E:\\climbing-aircraft-dataset\\global_min_max.json', 'r') as f:
            global_min_max = json.load(f)
        logging.info("全局最值已成功加载。")
    except Exception as e:
        logging.error(f"加载全局最值时发生错误: {e}")
        return

    # 准备处理的文件路径
    tasks = []
    for file in os.listdir(base_dir):
        if file.endswith('.csv'):
            file_path = os.path.join(base_dir, file)
            output_path = os.path.join(output_dir, f'normalized_{file}')
            tasks.append((file_path, output_path))
    logging.info(f"找到 {len(tasks)} 个待处理的文件。")

    # 使用并行化进行归一化处理
    with ThreadPoolExecutor(max_workers=4) as executor:  # 设置线程数，可以根据 CPU 核心数调整
        futures = [
            executor.submit(normalize_data_with_global_min_max, file_path, output_path, global_min_max)
            for file_path, output_path in tasks
        ]

        # 等待所有任务完成
        for future in futures:
            try:
                future.result()  # 如果任务中有异常，会在这里抛出
            except Exception as e:
                logging.error(f"任务执行时发生错误: {e}")

    logging.info("所有文件的数据归一化处理完成。")

if __name__ == '__main__':
    main()
