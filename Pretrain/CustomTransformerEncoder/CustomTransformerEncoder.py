import torch
import torch.nn as nn

from Pretrain.PretrainModel import DynamicGraphLearner, GCN


class CustomTransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, num_layers, nhead, dropout=0.1, seq_len=60, patch_size=1):
        super(CustomTransformerEncoder, self).__init__()

        self.dynamic_graph = DynamicGraphLearner(hidden_dim, hidden_dim * 2, hidden_dim, seq_len // patch_size)
        self.gcn = GCN(hidden_dim, hidden_dim * 2, hidden_dim, dropout)
        self.encoderLayer = nn.TransformerEncoderLayer(hidden_dim, nhead=nhead, dim_feedforward=2 * hidden_dim,
                                                       batch_first=True)

    def forward(self, x, mask=None):

        Tmask = ~mask

        encoder_output = self.encoderLayer(x, src_key_padding_mask=Tmask) * mask.unsqueeze(-1)

        # 动态图学习
        edge_features, edge_weights = self.dynamic_graph(encoder_output, mask)

        # 图卷积网络
        gcn_output = self.gcn(edge_features, edge_weights) * mask.unsqueeze(-1)

        combined_output = encoder_output + gcn_output

        return combined_output

# # 使用自定义的 Transformer 编码器
# class MyModel(nn.Module):
#     def __init__(self, hidden_dim, num_layers, nhead):
#         super(MyModel, self).__init__()
#         self.encoder = CustomTransformerEncoder(hidden_dim, num_layers, nhead)
#
#     def forward(self, x):
#         layer_outputs = self.encoder(x)
#         return layer_outputs
#
# # 示例用法
# model = MyModel(hidden_dim=64, num_layers=4, nhead=8)
# input_tensor = torch.randn(32, 60, 64)  # 示例输入(batch_size, seq_len, feature_dim)
# outputs = model(input_tensor)
#
# # outputs 现在包含每一层的输出
