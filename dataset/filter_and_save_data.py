import pandas as pd
import os

# 指定要保留的特征
desired_columns = ['segment', 'timestep', 'lat', 'lon', 'baroaltitudekalman',
                   'velocity', 'vertratecorr', 'taskalman', 'heading',
                   'ukalman', 'vkalman', 'tempkalman']


def filter_and_save_data(file_path, output_path):
    # 加载数据
    data = pd.read_csv(file_path)

    # 只保留所需的特征
    filtered_data = data[desired_columns]

    # 保存到新文件
    filtered_data.to_csv(output_path, index=False)


# 主程序
base_dir = 'E:\\climbing-aircraft-dataset'  # 数据所在目录
output_dir = 'E:\\climbing-aircraft-dataset\\filter_and_save_data'  # 输出目录
os.makedirs(output_dir, exist_ok=True)  # 创建输出目录

# 处理每个文件
for file in os.listdir(base_dir):
    if file.endswith('.csv'):
        file_path = os.path.join(base_dir, file)
        output_path = os.path.join(output_dir, f'filtered_{file}')

        # 输出处理日志
        print(f"正在处理文件: {file_path}")
        filter_and_save_data(file_path, output_path)
        print(f"已保存: {output_path}")

print("数据处理完成，已保存所需特征。")
