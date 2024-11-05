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