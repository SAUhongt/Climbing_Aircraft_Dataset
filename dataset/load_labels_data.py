import torch
from torch.utils.data import DataLoader, TensorDataset


def load_processed_data(file_path, batch_size=32):
    """
    从保存的.pt文件中加载处理后的特征和掩码。
    """
    data = torch.load(file_path)
    features = data['features']  # Shape: (num_samples, window_size, num_features)
    labels = data['labels']  # Shape: (num_samples, window_size, num_features)
    masks_input = data['masks_input']  # Shape: (num_samples, window_size)
    masks_label = data['masks_label']  # Shape: (num_samples, window_size)

    # 创建TensorDataset
    dataset = TensorDataset(features, labels, masks_input, masks_label)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# 示例使用
output_file_path = 'E:\\climbing-aircraft-dataset\\processed_data_with_labels.pt'
batch_size = 32

dataloader = load_processed_data(output_file_path, batch_size)

# 迭代数据
for features, label, mask_input, mask_label in dataloader:
    # 将 features 和 masks 送入神经网络进行处理
    pass
