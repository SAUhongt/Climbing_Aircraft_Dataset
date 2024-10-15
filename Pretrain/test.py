import torch
from tqdm import tqdm

from Pretrain.PretrainModel import PretrainModel
from Pretrain.train import generate_mask, compute_loss, feature_columns, seq_len, num_layers, dropout, patch_size, \
    logger, hidden_dim
from dataset.load_processed_data import load_processed_data


def test(model, test_loader, device):
    """
    使用测试集评估模型。

    Args:
        model: 已训练好的模型。
        test_loader: 测试集数据加载器。
        device: 模型运行的设备（CPU 或 GPU）。

    Returns:
        float: 测试集上的平均损失。
    """
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        with tqdm(test_loader, desc="Testing", unit="batch") as pbar:
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
                pbar.set_postfix(loss=running_loss / ((pbar.n + 1) * test_loader.batch_size))

    return running_loss / len(test_loader.dataset)

# 测试集路径和数据加载器
pretraining_data_test_path = 'E:\\climbing-aircraft-dataset\\dataTest\\pretraining_data_test.pt'
batch_size = 256
test_loader = load_processed_data(pretraining_data_test_path, batch_size)

# 加载训练好的模型和设备设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PretrainModel(feature_dim=len(feature_columns), seq_len=seq_len, hidden_dim=hidden_dim,
                      num_layers=num_layers, dropout=dropout, patch_size=patch_size, device=device)
model.load_state_dict(torch.load('best_pretrain_model/best_pretrain_model.pth'))
logger.info("loaded model: best_pretrain_model.pth")
model.to(device)

# 使用测试集进行评估
test_loss = test(model, test_loader, device)
logger.info(f"Test Loss: {test_loss:.4f}")
