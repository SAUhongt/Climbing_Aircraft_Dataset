import pandas as pd
import numpy as np
import torch

# 滑动窗口函数
def sliding_window_with_labels(data, window_size, input_size, label_size):
    """
    对单个轨迹数据应用滑动窗口，包含输入和标签。
    """
    if len(data) < window_size:
        # 如果数据长度小于窗口大小，按3:1划分输入和标签
        split_idx = len(data) * 3 // 4  # 3:1的划分点
        input_data = data[:split_idx]
        label_data = data[split_idx:]

        # 计算输入和标签的填充长度
        pad_input_length = input_size - len(input_data)
        pad_label_length = label_size - len(label_data)

        # 对输入和标签分别进行零填充
        padded_input = np.pad(input_data, ((0, pad_input_length), (0, 0)), mode='constant', constant_values=0)
        padded_label = np.pad(label_data, ((0, pad_label_length), (0, 0)), mode='constant', constant_values=0)

        # 生成掩码
        input_mask = np.zeros(input_size, dtype=bool)
        input_mask[:len(input_data)] = True  # 标记有效部分

        label_mask = np.zeros(label_size, dtype=bool)
        label_mask[:len(label_data)] = True  # 标记有效部分

        return [(padded_input, padded_label, input_mask, label_mask)]

    # 否则，正常应用滑动窗口
    windows = [(data[i:i + input_size], data[i + input_size:i + window_size], np.ones(input_size, dtype=bool),
                np.ones(label_size, dtype=bool))
               for i in range(0, len(data) - window_size + 1)]
    return windows


def process_and_save(file_path, output_path, window_size=80, input_size=60, label_size=20):
    """
    读取CSV数据，按segment分组并应用滑动窗口，将处理后的数据和掩码保存为.pt文件。
    """
    # 读取数据
    data = pd.read_csv(file_path)
    processed_data = []
    processed_labels = []
    processed_masks_input = []
    processed_masks_label = []

    sum = 0

    # 按segment分组并排序
    for segment_id, group in data.groupby('segment'):
        group = group.sort_values('timestep')

        # 转换为numpy数组并去掉非特征列
        group_features = group.drop(
            columns=['segment', 'timestep', 'lat_diff_normalized', 'lon_diff_normalized']).values

        # 对轨迹应用滑动窗口
        windows = sliding_window_with_labels(group_features, window_size, input_size, label_size)

        for input_data, label_data, input_mask, label_mask in windows:
            # 跳过输入或标签有效长度为 0 的窗口
            if not input_mask.any() or not label_mask.any():
                sum = sum + 1
                print(f"无效数据+1={sum},ADSB_len = {len(group_features)},input_len = {input_mask.sum()},"
                      f"label_len = {label_mask.sum()}")
                continue

            processed_data.append(input_data)
            processed_labels.append(label_data)
            processed_masks_input.append(input_mask)
            processed_masks_label.append(label_mask)

    # 将列表转换为numpy数组
    features_array = np.array(processed_data, dtype=np.float32)
    labels_array = np.array(processed_labels, dtype=np.float32)
    masks_input_array = np.array(processed_masks_input, dtype=np.bool_)
    masks_label_array = np.array(processed_masks_label, dtype=np.bool_)

    # 转换为torch张量
    features_tensor = torch.tensor(features_array)
    labels_tensor = torch.tensor(labels_array)
    masks_input_tensor = torch.tensor(masks_input_array)
    masks_label_tensor = torch.tensor(masks_label_array)

    # 保存成PyTorch格式
    torch.save({
        'features': features_tensor,
        'labels': labels_tensor,
        'masks_input': masks_input_tensor,
        'masks_label': masks_label_tensor
    }, output_path)
    print(features_array.shape)
    print(f'Saved processed data to {output_path}')


# 示例使用
window_size = 80
input_size = 60
label_size = 20

# input_file_path = 'E:\\climbing-aircraft-dataset\\dataTest\\Downstream_tasks_test.csv'  # 替换为你的CSV文件路径
# output_file_path = 'E:\\climbing-aircraft-dataset\\dataTest\\Downstream_tasks_test.pt'  # 指定保存的文件名
# process_and_save(input_file_path, output_file_path, window_size, input_size, label_size)
#
# input_file_path = 'E:\\climbing-aircraft-dataset\\dataTest\\Downstream_tasks_valid.csv'  # 替换为你的CSV文件路径
# output_file_path = 'E:\\climbing-aircraft-dataset\\dataTest\\Downstream_tasks_valid.pt'  # 指定保存的文件名
# process_and_save(input_file_path, output_file_path, window_size, input_size, label_size)
#
# input_file_path = 'E:\\climbing-aircraft-dataset\\dataTest\\Downstream_tasks_train.csv'  # 替换为你的CSV文件路径
# output_file_path = 'E:\\climbing-aircraft-dataset\\dataTest\\Downstream_tasks_train.pt'  # 指定保存的文件名
# process_and_save(input_file_path, output_file_path, window_size, input_size, label_size)

input_file_path = 'E:\\climbing-aircraft-dataset\\dataTest\\A319_tasks.csv'  # 替换为你的CSV文件路径
output_file_path = 'E:\\climbing-aircraft-dataset\\dataTest\\A319_tasks.pt'  # 指定保存的文件名
process_and_save(input_file_path, output_file_path, window_size, input_size, label_size)
