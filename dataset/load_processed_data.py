import torch
from torch.utils.data import DataLoader, TensorDataset


def load_processed_data(file_path, batch_size=32):
    """
    从保存的.pt文件中加载处理后的特征和掩码。
    """
    data = torch.load(file_path)
    features = data['features']  # Shape: (num_samples, window_size, num_features)
    masks = data['masks']  # Shape: (num_samples, window_size)

    # 创建TensorDataset
    dataset = TensorDataset(features, masks)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# 示例使用
output_file_path = 'E:\\climbing-aircraft-dataset\\processed_data.pt'
batch_size = 32

dataloader = load_processed_data(output_file_path, batch_size)

# 迭代数据
for features, masks in dataloader:
    # 将 features 和 masks 送入神经网络进行处理
    pass
