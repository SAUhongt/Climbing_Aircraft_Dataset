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

        # 统计每个文件中的空缺值情况
        missing_data = df.isnull().sum()

        # 计算每个数值列的最大值和最小值
        max_values = df.max(numeric_only=True)
        min_values = df.min(numeric_only=True)

        # 将统计信息保存到 summary_data 列表
        summary_data.append({
            'filename': filename,
            'missing_values': missing_data.to_dict(),  # 空缺值统计
            'max_values': max_values.to_dict(),  # 最大值
            'min_values': min_values.to_dict()  # 最小值
        })

        # 输出到控制台
        print(f"文件 {filename} 处理完成。")
        print(f"空缺值情况: \n{missing_data}")
        print(f"最大值: \n{max_values}")
        print(f"最小值: \n{min_values}")
        print("-" * 50)

        processed_count += 1
        print(f"已处理文件数: {processed_count}/{total_files}\n")

# 将所有文件的统计信息保存为 CSV 文件
if summary_data:
    # 为了便于展示，将统计结果转换为 DataFrame，并展开字典字段
    results = []
    for data in summary_data:
        for col in data['missing_values'].keys():
            results.append({
                'filename': data['filename'],
                'feature': col,
                'missing_values': data['missing_values'][col],
                'max_value': data['max_values'].get(col, 'N/A'),
                'min_value': data['min_values'].get(col, 'N/A')
            })

    summary_df = pd.DataFrame(results)

    # 导出为 CSV 文件
    output_file = os.path.join(output_folder, 'feature_statistics.csv')
    summary_df.to_csv(output_file, index=False)

    print(f"所有文件处理完成，结果已保存为 {output_file}")
else:
    print("没有成功处理的航班数据，无法导出结果。")
