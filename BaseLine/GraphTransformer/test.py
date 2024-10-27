import warnings

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from BaseLine.GraphTransformer.GraphTransformer import GraphTransformerBaseline
from BaseLine.LSTM.LSTM import LSTMBaseline
from BaseLine.Transformer.Transformer import TransformerBaseline
from Downstream_tasks.DownstreamModel import DownstreamModel
from Pretrain.PretrainModel import PretrainModel
from Pretrain.train import feature_columns, seq_len, num_layers, dropout, patch_size, logger, hidden_dim
from dataset.load_labels_data import load_processed_data

# 隐藏所有警告
warnings.filterwarnings("ignore")


def test_downstream(model, test_loader, device):
    model.eval()
    running_loss = 0.0
    all_outputs = []
    all_labels = []
    all_valid_masks = []

    with torch.no_grad():
        with tqdm(test_loader, desc="Testing", unit="batch") as pbar:
            for features, labels, masks_input, masks_label in pbar:
                features, labels, masks_input, masks_label = (
                    features.to(device),
                    labels.to(device),
                    masks_input.to(device),
                    masks_label.to(device)
                )

                # 模型进行预测
                outputs = model(features, masks_input, masks_label)

                # 计算损失，仅计算有效区域的损失
                loss = nn.MSELoss()(outputs * masks_label.unsqueeze(-1), labels * masks_label.unsqueeze(-1))
                running_loss += loss.item() * features.size(0)
                pbar.set_postfix(loss=running_loss / ((pbar.n + 1) * test_loader.batch_size))

                # 收集所有预测和标签
                all_outputs.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_valid_masks.append(masks_label.cpu().numpy())

    # 合并所有输出和标签
    all_outputs = np.concatenate(all_outputs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_valid_masks = np.concatenate(all_valid_masks, axis=0).astype(bool)

    # 处理有效掩码
    valid_mask = all_valid_masks.flatten()  # 扁平化掩码

    # 展平输出和标签
    all_outputs_flat = all_outputs.reshape(-1, all_outputs.shape[-1])
    all_labels_flat = all_labels.reshape(-1, all_labels.shape[-1])

    # 应用掩码
    all_outputs_valid = all_outputs_flat[valid_mask]
    all_labels_valid = all_labels_flat[valid_mask]

    # 计算总的指标
    rmse = mean_squared_error(all_labels_valid, all_outputs_valid, squared=False)
    mae = mean_absolute_error(all_labels_valid, all_outputs_valid)
    r2 = r2_score(all_labels_valid, all_outputs_valid)

    # 针对前六个特征计算指标
    metrics = {}
    for i in range(len(feature_columns)):
        rmse_i = mean_squared_error(all_labels_valid[:, i], all_outputs_valid[:, i], squared=False)
        mae_i = mean_absolute_error(all_labels_valid[:, i], all_outputs_valid[:, i])
        r2_i = r2_score(all_labels_valid[:, i], all_outputs_valid[:, i])

        # metrics[f'Feature_{i+1}'] = {
        #     'RMSE': rmse_i,
        #     'MAE': mae_i,
        #     'R^2': r2_i
        # }

        metrics[f'{feature_columns[i]}'] = {
            'RMSE': rmse_i,
            'MAE': mae_i,
            'R^2': r2_i
        }

    return running_loss / len(test_loader.dataset), rmse, mae, r2, metrics

# 测试集路径和数据加载器
downstream_data_test_path = 'E:\\climbing-aircraft-dataset\\dataTest\\Downstream_tasks_test.pt'
batch_size = 256
test_loader = load_processed_data(downstream_data_test_path, batch_size)

# 加载训练好的模型和设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

graphTransformer_model = GraphTransformerBaseline(len(feature_columns), hidden_dim, num_layers, dropout=dropout).to(device)

graphTransformer_model.load_state_dict(torch.load('best_graphTransformer_model/best_graphTransformer_model.pth'))
logger.info("Loaded model: best_graphTransformer_model/best_graphTransformer_model.pth")
graphTransformer_model.to(device)

# 使用测试集进行评估


test_loss, rmse, mae, r2, metrics = test_downstream(graphTransformer_model, test_loader, device)

logger.info(f"Test Loss: {test_loss:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, R^2: {r2:.6f}")

# 打印各个指标
for feature, metric in metrics.items():
    logger.info(f"{feature}: RMSE: {metric['RMSE']:.6f}, MAE: {metric['MAE']:.6f}, R^2: {metric['R^2']:.6f}")
