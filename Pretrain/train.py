import os
import random

import numpy as np
import torch
from torch import nn
import logging
from tqdm import tqdm  # 引入 tqdm
from Pretrain.PretrainModel import PretrainModel
from dataset.load_processed_data import load_processed_data

import warnings
warnings.filterwarnings("ignore", category=UserWarning,
                        message="The PyTorch API of nested tensors is in prototype stage")

# 设置日志记录
log_filename = "log/training_log_base.log"  # 使用固定的日志文件名，这样可以在重新训练时继续记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 控制台输出
        logging.FileHandler(log_filename, mode='a')  # 追加模式下保存日志
    ]
)
logger = logging.getLogger(__name__)



def set_random_seed(seed=42):
    """
    设置全局随机种子以确保结果的可复现性。

    Args:
        seed (int): 要使用的随机种子。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    torch.backends.cudnn.benchmark = False  # 设置为False可以提高可重复性

def generate_mask(sequences, valid_mask, drop_prob=0.2, min_span=5, max_span=10, max_drops=3):
    """
    生成一个掩码矩阵，用于模拟缺失值填充，支持对单个特征的连续抹除。
    返回的 masks 是一个二进制矩阵，并通过增加抹除的概率、跨度和次数来增加填充难度。

    Args:
        sequences (Tensor): 输入序列，形状为 (batch_size, seq_len, feature_dim)。
        valid_mask (Tensor): 有效掩码，形状为 (batch_size, seq_len)。
        drop_prob (float): 每个特征被抹除的概率。
        min_span (int): 抹除的最小跨度。
        max_span (int): 抹除的最大跨度。
        max_drops (int): 每个特征可能被抹除的最大区间数量。

    Returns:
        masked_sequences (Tensor): 掩盖后的序列。
        masks (Tensor): 二进制掩码矩阵。
    """
    batch_size, seq_len, feature_dim = sequences.shape
    masks = torch.ones_like(sequences)  # 初始化全1矩阵
    valid_mask_expanded = valid_mask.unsqueeze(-1).expand(-1, -1, feature_dim)

    for i in range(batch_size):
        for j in range(feature_dim):
            if torch.rand(1).item() < drop_prob:
                valid_indices = torch.nonzero(valid_mask[i], as_tuple=True)[0]
                if len(valid_indices) == 0:
                    continue

                # 随机选择最多 max_drops 个区间进行抹除
                num_drops = torch.randint(1, max_drops + 1, (1,)).item()
                for _ in range(num_drops):
                    start_idx = valid_indices[torch.randint(0, len(valid_indices), (1,)).item()].item()
                    span_len = torch.randint(min_span, max_span + 1, (1,)).item()
                    end_idx = min(start_idx + span_len, valid_indices[-1] + 1)
                    masks[i, start_idx:end_idx, j] = 0  # 设置连续抹除的区间为0

    # 保证返回的 masked_sequences 和 masks 都是 0 和 1 组成的二进制矩阵
    masked_sequences = sequences * masks * valid_mask_expanded
    masks = masks * valid_mask_expanded  # 确保掩码也仅在有效区域起作用

    return masked_sequences, masks.bool()


def compute_loss(outputs, sequences, fea_masks, valid_mask_expanded):
    """
    计算损失：仅计算被抹除部分的损失
    """
    Nfea_masks = ~ fea_masks
    outputs = outputs.view(-1, outputs.shape[-1])
    sequences = sequences.view(-1, sequences.shape[-1])
    Nfea_masks = Nfea_masks.view(-1, fea_masks.shape[-1])
    valid_mask_expanded = valid_mask_expanded.view(-1, valid_mask_expanded.shape[-1])

    # 仅计算被抹除部分的损失
    # l1 = nn.MSELoss()(outputs * Nfea_masks * valid_mask_expanded, sequences * Nfea_masks * valid_mask_expanded)
    # 计算所有有效区域的损失
    l2 = nn.MSELoss()(outputs * valid_mask_expanded, sequences * valid_mask_expanded)

    return l2


def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    running_loss = 0.0
    # 使用 tqdm 包装 train_loader，显示进度条
    with tqdm(train_loader, desc="Training", unit="batch") as pbar:
        for sequences, valid_mask in pbar:
            sequences, valid_mask = sequences.to(device), valid_mask.to(device)
            Tmasks = ~valid_mask
            masked_sequences, fea_masks = generate_mask(sequences, valid_mask, drop_prob=0.8, min_span=2, max_span=10,
                                                        max_drops=4)

            optimizer.zero_grad()
            outputs = model(masked_sequences, mask=Tmasks, is_pretraining=True)
            loss = compute_loss(outputs, sequences, fea_masks, valid_mask.unsqueeze(-1).expand(-1, -1, sequences.size(-1)))

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * sequences.size(0)

            # 更新进度条中的损失信息
            pbar.set_postfix(loss=running_loss / ((pbar.n + 1) * train_loader.batch_size))

    return running_loss / len(train_loader.dataset)


def validate(model, valid_loader, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        with tqdm(valid_loader, desc="Validating", unit="batch") as pbar:
            for sequences, valid_mask in pbar:
                sequences, valid_mask = sequences.to(device), valid_mask.to(device)
                Tmasks = ~valid_mask
                masked_sequences, fea_masks = generate_mask(sequences, valid_mask, drop_prob=0.8, min_span=2,
                                                            max_span=10, max_drops=4)

                outputs = model(masked_sequences, mask=Tmasks, is_pretraining=True)
                loss = compute_loss(outputs, sequences, fea_masks,
                                    valid_mask.unsqueeze(-1).expand(-1, -1, sequences.size(-1)))

                running_loss += loss.item() * sequences.size(0)

                # 更新进度条中的损失信息
                pbar.set_postfix(loss=running_loss / ((pbar.n + 1) * valid_loader.batch_size))

    return running_loss / len(valid_loader.dataset)


def save_checkpoint(model, optimizer, epoch, best_valid_loss, checkpoint_path='checkpoint/checkpoint.pth'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_valid_loss': best_valid_loss
    }, checkpoint_path)
    logger.info(f"saved model: {checkpoint_path}")
    logger.info(f"Checkpoint saved at epoch {epoch + 1}")


def load_checkpoint(model, optimizer, checkpoint_path='checkpoint/checkpoint.pth'):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_valid_loss = checkpoint['best_valid_loss']
        logger.info(f"loaded model: {checkpoint_path}")
        logger.info(f"Checkpoint loaded from epoch {epoch + 1}")
        return epoch, best_valid_loss
    else:
        logger.info("No checkpoint found. Starting training from scratch.")
        return 0, float('inf')



# 设置随机种子
set_random_seed(seed=42)

# 模型配置
feature_columns = ['lat_diff_normalized', 'lon_diff_normalized', 'x_normalized', 'y_normalized',
                   'baroaltitudekalman_normalized', 'velocity_normalized', 'vertratecorr_normalized',
                   'taskalman_normalized', 'heading_normalized', 'ukalman_normalized', 'vkalman_normalized',
                   'tempkalman_normalized']
seq_len = 60
hidden_dim = 64
num_layers = 4
dropout = 0.5
patch_size = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train():
    # 创建模型
    model = PretrainModel(feature_dim=len(feature_columns), seq_len=seq_len, hidden_dim=hidden_dim,
                          num_layers=num_layers, dropout=dropout, patch_size=patch_size, device=device)

    # 损失函数和优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 数据加载
    pretraining_data_path = 'E:\\climbing-aircraft-dataset\\dataTest\\pretraining_data_train.pt'
    pretraining_data_valid_path = 'E:\\climbing-aircraft-dataset\\dataTest\\pretraining_data_valid.pt'
    batch_size = 256
    train_loader = load_processed_data(pretraining_data_path, batch_size)
    valid_loader = load_processed_data(pretraining_data_valid_path, batch_size)

    # 加载检查点
    start_epoch, best_valid_loss = load_checkpoint(model, optimizer)

    # 训练模型
    num_epochs = 50
    patience = 5
    early_stop_counter = 0

    for epoch in range(start_epoch, num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        valid_loss = validate(model, valid_loader, device)

        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best_pretrain_model/best_pretrain_model.pth')
            logger.info(f"Saved best model at epoch {epoch + 1}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            logger.info(f"No improvement for {early_stop_counter} epochs.")

        # 保存最新检查点
        save_checkpoint(model, optimizer, epoch, best_valid_loss)

        if early_stop_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break


if __name__ == '__main__':
    train()


# # 加载最佳模型
# model.load_state_dict(torch.load('best_pretrain_model/best_pretrain_model.pth'))
# model.to(device)
