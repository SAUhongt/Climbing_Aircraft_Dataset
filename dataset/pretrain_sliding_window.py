import pandas as pd
import numpy as np
import torch


# 滑动窗口函数
def sliding_window(data, window_size, step_size):
    """
    对单个轨迹数据应用滑动窗口。
    """
    if len(data) < window_size:
        # 如果数据长度小于窗口大小，直接返回一个填充的窗口
        pad_length = window_size - len(data)
        padded_data = np.pad(data, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)

        # 创建掩码，标记填充部分
        mask = np.ones(window_size, dtype=bool)
        mask[len(data):] = False  # 标记补零部分为False

        return [(padded_data, mask)]  # 返回数据和掩码的元组

    # 否则，正常应用滑动窗口
    num_windows = (len(data) - window_size) // step_size + 1
    windows = []
    for i in range(0, len(data) - window_size + 1, step_size):
        window = data[i:i + window_size]
        mask = np.ones(window_size, dtype=bool)  # 全部标记为True
        windows.append((window, mask))  # 返回数据和掩码的元组

    return windows


def process_and_save(file_path, output_path, window_size=60, step_size=1):
    """
    读取CSV数据，按segment分组并应用滑动窗口，将处理后的数据和掩码保存为.pt文件。
    """
    # 读取数据
    data = pd.read_csv(file_path)
    processed_data = []
    processed_masks = []

    # 按segment分组并排序
    for segment_id, group in data.groupby('segment'):
        group = group.sort_values('timestep')

        # 转换为numpy数组并去掉非特征列
        group_features = group.drop(
            columns=['segment', 'timestep', 'lat_diff_normalized', 'lon_diff_normalized']).values

        # 对轨迹应用滑动窗口
        windows = sliding_window(group_features, window_size, step_size)

        for window, mask in windows:
            processed_data.append(window)
            processed_masks.append(mask)

    # 转换为torch张量并保存
    features_tensor = torch.tensor(np.array(processed_data), dtype=torch.float32)  # 优化
    masks_tensor = torch.tensor(np.array(processed_masks), dtype=torch.bool)  # 优化

    # 保存成PyTorch格式
    torch.save({'features': features_tensor, 'masks': masks_tensor}, output_path)
    print(f'Saved processed data to {output_path}')

window_size = 60
step_size = 5


#调试维度用的一个小文件
input_file_path = 'E:\\climbing-aircraft-dataset\\dataTest\\normalized_diff_filtered_A319_train.csv'  # 替换为你的CSV文件路径
output_file_path = 'E:\\climbing-aircraft-dataset\\dataTest\\pretraining_data_train.pt'  # 指定保存的文件名
process_and_save(input_file_path, output_file_path, window_size, step_size)
input_file_path = 'E:\\climbing-aircraft-dataset\\dataTest\\normalized_diff_filtered_A319_valid.csv'  # 替换为你的CSV文件路径
output_file_path = 'E:\\climbing-aircraft-dataset\\dataTest\\pretraining_data_valid.pt'  # 指定保存的文件名
process_and_save(input_file_path, output_file_path, window_size, step_size)
input_file_path = 'E:\\climbing-aircraft-dataset\\dataTest\\normalized_diff_filtered_A319_test.csv'  # 替换为你的CSV文件路径
output_file_path = 'E:\\climbing-aircraft-dataset\\dataTest\\pretraining_data_test.pt'  # 指定保存的文件名
process_and_save(input_file_path, output_file_path, window_size, step_size)




# 示例使用
# input_file_path = 'E:\\climbing-aircraft-dataset\\pretraining_data\\pretraining_data.csv'  # 替换为你的CSV文件路径
# output_file_path = 'E:\\climbing-aircraft-dataset\\pretraining_data\\pretraining_data.pt'  # 指定保存的文件名
# process_and_save(input_file_path, output_file_path, window_size, step_size)
# input_file_path = 'E:\\climbing-aircraft-dataset\\downstream_data\\normalized_valid.csv'  # 替换为你的CSV文件路径
# output_file_path = 'E:\\climbing-aircraft-dataset\\pretraining_data\\pretraining_data_valid.pt'  # 指定保存的文件名
# process_and_save(input_file_path, output_file_path, window_size, step_size)
