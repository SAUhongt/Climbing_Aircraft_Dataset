import torch
from torch import nn

from Pretrain.TCN.tcn import TemporalConvNet


class DownstreamModel(nn.Module):
    def __init__(self, pretrain_model, feature_dim, hidden_dim, num_layers, dropout):
        super(DownstreamModel, self).__init__()
        self.pretrain_model = pretrain_model
        self.pretrain_model.requires_grad_(False)  # 冻结预训练模型的参数

        self.tcn = TemporalConvNet(hidden_dim, [hidden_dim], kernel_size=3)

        # 使用 LSTM 作为预测头
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, nhead=8, dim_feedforward=2 * hidden_dim, batch_first=True),
            num_layers=num_layers
        )
        self.linear = nn.Linear(hidden_dim, feature_dim)

        # 输出层，将 LSTM 的输出映射到原始特征维度
        self.output_layer = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x, mask=None, output_length=20):
        # 使用预训练模型生成的上下文表示
        combined_output = self.pretrain_model(x, mask, is_pretraining=False)

        # combined_output 形状是 (batch_size, seq_len, hidden_dim)
        # 使用 LSTM 进行进一步的序列建模
        # lstm_output, _ = self.lstm(combined_output)

        transformer_output = self.encoder(combined_output)


        # 根据 output_length 确定输出的长度
        if output_length is not None:
            # 使用切片选择输出的长度
            output = self.output_layer(transformer_output[:, :output_length, :])
        else:
            # 如果未指定输出长度，则使用整个输出
            output = self.output_layer(transformer_output)

        # output 形状为 (batch_size, output_length, feature_dim)
        return output

