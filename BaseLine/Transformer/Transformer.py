import torch
import torch.nn as nn

from Pretrain.PretrainModel import PatchEmbedding, PositionalEncoding


class TransformerBaseline(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers=1, dropout=0.1, seq_len=60, patch_size=1):
        super(TransformerBaseline, self).__init__()
        self.hidden_size = hidden_dim
        self.num_layers = num_layers

        self.patch_embedding = PatchEmbedding(feature_dim, hidden_dim, patch_size)
        self.positional_encoding = PositionalEncoding(hidden_dim, seq_len)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, nhead=8, dim_feedforward=2 * hidden_dim, batch_first=True),
            num_layers=num_layers
        )
        self.linear = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x, input_mask, output_mask):
        """
        x: 输入序列, shape = (batch_size, input_seq_length, input_size)
        input_mask: 输入序列掩码, shape = (batch_size, input_seq_length)
        output_mask: 输出序列掩码, shape = (batch_size, output_seq_length)
        """

        _, target_len = output_mask.size()

        Tmask = ~input_mask

        x = self.patch_embedding(x)
        x = self.positional_encoding(x)
        batch_size, seq_length, feature_dim = x.size()

        encoder_output = self.encoder(
            x, src_key_padding_mask=Tmask.view(batch_size, -1) if Tmask is not None else None
        ) * input_mask.unsqueeze(-1)

        output = self.linear(encoder_output)

        # 只取 output_mask 指定的长度
        output = output[:, :target_len, :] * output_mask.unsqueeze(-1)

        return output

# # 示例参数
# input_size = 10  # 输入特征数量
# hidden_size = 64  # LSTM 隐藏层大小
# output_size = 1  # 预测输出特征大小
# num_layers = 2  # LSTM 层数
#
# # 实例化模型
# model = LSTMBaseline(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
#
# # 示例输入
# batch_size = 32
# input_seq_length = 60  # 输入序列长度
# output_seq_length = 20  # 输出序列长度
# x = torch.randn(batch_size, input_seq_length, input_size)
# input_mask = torch.randint(0, 2, (batch_size, input_seq_length)).float()  # 随机生成输入掩码
# output_mask = torch.randint(0, 2, (batch_size, output_seq_length)).float()  # 随机生成输出掩码
#
# # 前向传播
# output = model(x, input_mask, output_mask)
# print("Output shape:", output.shape)  # 应为 (batch_size, output_seq_length, output_size)
