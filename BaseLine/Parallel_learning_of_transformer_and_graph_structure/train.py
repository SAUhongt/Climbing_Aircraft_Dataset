import os
import random
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from BaseLine.Parallel_learning_of_transformer_and_graph_structure.GraphTransformer import GraphTransformerBaseline
from Pretrain.train import feature_columns, seq_len, hidden_dim, patch_size, dropout, num_layers, logger, target_len
from dataset.load_labels_data import load_processed_data
import warnings

warnings.filterwarnings("ignore", category=UserWarning,
                        message="The PyTorch API of nested tensors is in prototype stage")

# 设置随机种子
def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 计算损失函数
def compute_loss(outputs, targets, valid_mask):
    """
    计算序列预测的损失，仅考虑有效区域。
    """
    loss = nn.MSELoss()(outputs * valid_mask, targets * valid_mask)
    return loss

# 单轮训练函数
def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    running_loss = 0.0
    with tqdm(train_loader, desc="Training", unit="batch") as pbar:
        for features, labels, masks_input, masks_label in pbar:
            features, labels, masks_input, masks_label = (
                features.to(device),
                labels.to(device),
                masks_input.to(device),
                masks_label.to(device)
            )

            optimizer.zero_grad()
            outputs = model(features, masks_input, masks_label)
            loss = compute_loss(outputs, labels, masks_label.unsqueeze(-1))

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * features.size(0)
            pbar.set_postfix(loss=running_loss / ((pbar.n + 1) * train_loader.batch_size))

    return running_loss / len(train_loader.dataset)

# 验证函数
def validate(model, valid_loader, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        with tqdm(valid_loader, desc="Validating", unit="batch") as pbar:
            for features, labels, masks_input, masks_label in pbar:
                features, labels, masks_input, masks_label = (
                    features.to(device),
                    labels.to(device),
                    masks_input.to(device),
                    masks_label.to(device)
                )

                outputs = model(features, masks_input, masks_label)
                loss = compute_loss(outputs, labels, masks_label.unsqueeze(-1))

                running_loss += loss.item() * features.size(0)
                pbar.set_postfix(loss=running_loss / ((pbar.n + 1) * valid_loader.batch_size))

    return running_loss / len(valid_loader.dataset)

# 保存检查点
def save_checkpoint(model, optimizer, epoch, best_valid_loss, checkpoint_path='checkpoint/downstream_checkpoint.pth'):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)  # 创建目录（如果不存在）
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_valid_loss': best_valid_loss
    }, checkpoint_path)
    logger.info(f"Saved model: {checkpoint_path}")
    logger.info(f"Checkpoint saved at epoch {epoch + 1}")

# 加载检查点
def load_checkpoint(model, optimizer, checkpoint_path='checkpoint/downstream_checkpoint.pth'):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_valid_loss = checkpoint['best_valid_loss']
        logger.info(f"Loaded model: {checkpoint_path}")
        logger.info(f"Checkpoint loaded from epoch {epoch + 1}")
        return epoch, best_valid_loss
    else:
        logger.info("No checkpoint found. Starting training from scratch.")
        return 0, float('inf')

# 训练函数
def train():
    # 加载预训练模型并构建下游任务模型
    graphTransformer_model = GraphTransformerBaseline(len(feature_columns), hidden_dim, num_layers, dropout=dropout).to(device)

    # 将模型结构输出到日志
    logger.info(f"Model structure: {graphTransformer_model}")

    # 定义损失函数和优化器
    optimizer = torch.optim.Adam(graphTransformer_model.parameters(), lr=1e-4)

    # 数据加载
    downstream_train_path = 'E:\\climbing-aircraft-dataset\\dataTest\\Downstream_tasks_train.pt'
    downstream_valid_path = 'E:\\climbing-aircraft-dataset\\dataTest\\Downstream_tasks_valid.pt'
    batch_size = 256
    train_loader = load_processed_data(downstream_train_path, batch_size)
    valid_loader = load_processed_data(downstream_valid_path, batch_size)

    # 加载检查点
    start_epoch, best_valid_loss = load_checkpoint(graphTransformer_model, optimizer)

    # 训练模型
    num_epochs = 200
    patience = 5
    early_stop_counter = 0

    for epoch in range(start_epoch, num_epochs):
        train_loss = train_one_epoch(graphTransformer_model, train_loader, optimizer, device)
        valid_loss = validate(graphTransformer_model, valid_loader, device)

        logger.info(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}, Valid Loss: {valid_loss:.6f}")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(graphTransformer_model.state_dict(), 'best_graphTransformer_model/best_graphTransformer_model.pth')
            logger.info(f"Saved best model at epoch {epoch + 1}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            logger.info(f"No improvement for {early_stop_counter} epochs.")

        # 保存最新检查点
        save_checkpoint(graphTransformer_model, optimizer, epoch, best_valid_loss)

        if early_stop_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

# 配置
set_random_seed(seed=42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    train()
