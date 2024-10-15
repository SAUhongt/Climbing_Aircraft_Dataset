import torch
from torch import nn


class DownstreamModel(nn.Module):
    def __init__(self, pretrain_model, feature_dim, hidden_dim, num_layers, dropout):
        super(DownstreamModel, self).__init__()
        self.pretrain_model = pretrain_model
        self.pretrain_model.requires_grad_(False)  # 冻结预训练模型的参数

        # 使用 LSTM 作为预测头
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # 输出层，将 LSTM 的输出映射到原始特征维度
        self.output_layer = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x, mask=None):
        # 使用预训练模型生成的上下文表示
        combined_output = self.pretrain_model(x, mask, is_pretraining=False)

        # combined_output 形状是 (batch_size, seq_len, hidden_dim)
        # 使用 LSTM 进行进一步的序列建模
        lstm_output, _ = self.lstm(combined_output)

        # 使用全连接层将 LSTM 的输出映射到 feature_dim 大小
        output = self.output_layer(lstm_output)

        # output 形状为 (batch_size, seq_len, feature_dim)
        return output
