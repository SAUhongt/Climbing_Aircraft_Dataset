import pandas as pd
import os

def merge_files(input_folder, output_file):
    """
    合并指定文件夹中的所有CSV文件，并保存为一个输出文件。
    :param input_folder: 包含多个CSV文件的文件夹路径。
    :param output_file: 合并后保存的输出文件路径。
    """
    all_data = []
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.csv'):
            file_path = os.path.join(input_folder, file_name)
            data = pd.read_csv(file_path)
            all_data.append(data)
            print(f"Loaded {file_name}")

    # 合并所有数据
    merged_data = pd.concat(all_data, ignore_index=True)
    # 保存合并后的数据
    merged_data.to_csv(output_file, index=False)
    print(f"Saved merged data to {output_file}")

# 示例使用
train_input_folder = 'E:\\climbing-aircraft-dataset\\downstream_data\\normalized_data\\test'  # 替换为训练集文件所在的文件夹路径
valid_input_folder = 'E:\\climbing-aircraft-dataset\\downstream_data\\normalized_data\\valid'  # 替换为验证集文件所在的文件夹路径
train_output_file = 'E:\\climbing-aircraft-dataset\\downstream_data\\normalized_train.csv'
valid_output_file = 'E:\\climbing-aircraft-dataset\\downstream_data\\normalized_valid.csv'


# 合并训练集和验证集
merge_files(train_input_folder, train_output_file)
merge_files(valid_input_folder, valid_output_file)
