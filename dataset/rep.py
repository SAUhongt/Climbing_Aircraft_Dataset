import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor

def remove_duplicate_lon(file_path):
    try:
        # 加载数据
        data = pd.read_csv(file_path)

        # 检查是否存在重复的 'lon' 列
        if 'lon.1' in data.columns:
            # 删除重复的 'lon' 列（通常是第二个出现的）
            data = data.loc[:, ~data.columns.duplicated()]
            print(f"'lon.1' 列已移除: {file_path}")

        # 保存修改后的数据，覆盖原文件
        data.to_csv(file_path, index=False)
        print(f"已保存并覆盖: {file_path}")
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")

# 主程序
base_dir = 'E:\\climbing-aircraft-dataset\\lat_lon_diff_data'  # 输入文件夹
max_workers = 4  # 设置并发线程数

# 获取所有CSV文件的路径
file_paths = [os.path.join(base_dir, file) for file in os.listdir(base_dir) if file.endswith('.csv')]

# 使用 ThreadPoolExecutor 并行处理文件
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    executor.map(remove_duplicate_lon, file_paths)

print("重复的 'lon' 列处理完成，已覆盖原文件。")
