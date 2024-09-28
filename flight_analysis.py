import pandas as pd
import os

# 文件夹路径，替换为你的 CSV 文件所在的路径
folder_path = 'E:\\climbing-aircraft-dataset'

# 输出文件夹路径，替换为你想保存结果的路径
output_folder = 'E:\\output-folder'  # 你需要将此路径替换为有效的输出路径
os.makedirs(output_folder, exist_ok=True)  # 如果输出文件夹不存在，则创建

# 统计文件数和处理进度
file_count = 0
processed_count = 0

# 获取文件夹中的 CSV 文件数量
total_files = len([f for f in os.listdir(folder_path) if f.endswith('.csv')])
print(f"找到 {total_files} 个 CSV 文件。开始处理...")

# 存储每个文件统计信息的列表
summary_data = []

# 遍历文件夹中的所有 CSV 文件
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_count += 1
        print(f"正在处理文件 {file_count}/{total_files}: {filename}")

        # 读取每个 CSV 文件
        filepath = os.path.join(folder_path, filename)
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            print(f"读取文件 {filename} 时出错: {e}")
            continue

        # 按 `segment` 分组，并统计每个航班的轨迹点数量（时间序列长度）
        flight_counts = df.groupby('segment').size()

        # 保存每个文件的统计信息到列表
        summary_data.append({
            'filename': filename,
            'flight_count': len(flight_counts),  # 航班数量
            'mean_trajectory_length': flight_counts.mean(),  # 每个航班的平均轨迹点长度
            'min_trajectory_length': flight_counts.min(),  # 每个航班的最小轨迹点长度
            'max_trajectory_length': flight_counts.max(),  # 每个航班的最大轨迹点长度
        })

        # 输出到控制台
        print(f"文件 {filename} 处理完成。")
        print(f"航班数量: {len(flight_counts)}")
        print(f"平均轨迹点长度: {flight_counts.mean()}")
        print(f"最小轨迹点长度: {flight_counts.min()}")
        print(f"最大轨迹点长度: {flight_counts.max()}")
        print("-" * 50)

        processed_count += 1
        print(f"已处理文件数: {processed_count}/{total_files}\n")

# 将所有文件的统计信息保存为 CSV 文件
if summary_data:
    summary_df = pd.DataFrame(summary_data)

    # 导出为 CSV 文件
    output_file = os.path.join(output_folder, 'flight_summary_results.csv')
    summary_df.to_csv(output_file, index=False)

    print(f"所有文件处理完成，结果已保存为 {output_file}")
else:
    print("没有成功处理的航班数据，无法导出结果。")
