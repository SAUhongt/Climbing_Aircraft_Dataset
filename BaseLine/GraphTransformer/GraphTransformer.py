import torch
import torch.nn as nn

from Pretrain.CustomTransformerEncoder.CustomTransformerEncoder import CustomTransformerEncoder
from Pretrain.PretrainModel import PatchEmbedding, PositionalEncoding, DynamicGraphLearner, GCN


class GraphTransformerBaseline(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers=1, dropout=0.1, seq_len=60, patch_size=1):
        super(GraphTransformerBaseline, self).__init__()
        self.hidden_size = hidden_dim
        self.num_layers = num_layers

        self.patch_embedding = PatchEmbedding(feature_dim, hidden_dim, patch_size)
        self.positional_encoding = PositionalEncoding(hidden_dim, seq_len)
        self.layers = nn.ModuleList([
            CustomTransformerEncoder(hidden_dim, num_layers, nhead=8, dropout=dropout, seq_len=seq_len,
                                     patch_size=patch_size)
            for _ in range(num_layers)
        ])
        self.linear = nn.Linear(hidden_dim, feature_dim)
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x, input_mask, output_mask):
        _, target_len = output_mask.size()

        x = self.patch_embedding(x)
        x = self.positional_encoding(x)

        for index, layer in enumerate(self.layers):
            # 当前层的输出
            layer_output = layer(x, input_mask) * input_mask.unsqueeze(-1)

            # 残差连接
            x = layer_output + x

            # 层归一化
            x = self.norms[index](x)

        output = self.linear(x)

        # 只取 output_mask 指定的长度
        output = output[:, :target_len, :] * output_mask.unsqueeze(-1)

        return output

