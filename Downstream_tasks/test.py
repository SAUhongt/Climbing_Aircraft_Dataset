import torch
from torch import nn
from tqdm import tqdm

from Downstream_tasks.DownstreamModel import DownstreamModel
from Pretrain.PretrainModel import PretrainModel
from Pretrain.train import feature_columns, seq_len, num_layers, dropout, patch_size, \
    logger, hidden_dim
from dataset.load_labels_data import load_processed_data


def test_downstream(model, test_loader, device):
    """
    使用测试集评估下游任务模型。

    Args:
        model: 已训练好的下游任务模型。
        test_loader: 测试集数据加载器。
        device: 模型运行的设备（CPU 或 GPU）。

    Returns:
        float: 测试集上的平均损失。
    """
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        with tqdm(test_loader, desc="Testing", unit="batch") as pbar:
            for features, labels, masks_input, masks_label in pbar:
                features, labels, masks_input, masks_label = (
                    features.to(device),
                    labels.to(device),
                    masks_input.to(device),
                    masks_label.to(device)
                )

                # 下游任务模型进行预测，is_pretraining 设置为 False
                outputs = model(features, mask=masks_input)

                # 计算损失，仅计算有效区域的损失
                loss = nn.MSELoss()(outputs * masks_label.unsqueeze(-1), labels * masks_label.unsqueeze(-1))

                running_loss += loss.item() * features.size(0)

                # 更新进度条中的损失信息
                pbar.set_postfix(loss=running_loss / ((pbar.n + 1) * test_loader.batch_size))

    return running_loss / len(test_loader.dataset)


# 测试集路径和数据加载器
downstream_data_test_path = 'E:\\climbing-aircraft-dataset\\dataTest\\Downstream_tasks_test.pt'
batch_size = 256
test_loader = load_processed_data(downstream_data_test_path, batch_size)

# 加载训练好的模型和设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



pretrain_model = PretrainModel(feature_dim=len(feature_columns), seq_len=seq_len, hidden_dim=hidden_dim,
                                   num_layers=num_layers, dropout=dropout, patch_size=patch_size, device=device)
pretrain_checkpoint = torch.load('../Pretrain/best_pretrain_model/best_pretrain_model.pth')
logger.info("Loaded model: ../Pretrain/best_pretrain_model/best_pretrain_model.pth")
pretrain_model.load_state_dict(pretrain_checkpoint)
downstream_model = DownstreamModel(
    pretrain_model=pretrain_model,
    feature_dim=len(feature_columns),
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    dropout=dropout
).to(device)


downstream_model.load_state_dict(torch.load('best_downstream_model/best_downstream_model.pth'))
logger.info("Loaded model: best_downstream_model/best_downstream_model.pth")
downstream_model.to(device)

# 使用测试集进行评估
test_loss = test_downstream(downstream_model, test_loader, device)
logger.info(f"Test Loss: {test_loss:.4f}")
