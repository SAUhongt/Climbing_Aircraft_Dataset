import torch
import torch.nn as nn

from BaseLine.Parallel_learning_of_transformer_and_graph_structure.Dilated_inception_Layer.Dilated_inception import \
    DilatedInception
from Pretrain.PretrainModel import DynamicGraphLearner, GCN


class CustomTransformerEncoder(nn.Module):
    def __init__(self, hidden_dim, nhead, dropout=0.1, seq_len=60, patch_size=1):
        super(CustomTransformerEncoder, self).__init__()

        self.dynamic_graph = DynamicGraphLearner(hidden_dim, hidden_dim * 2, hidden_dim, seq_len // patch_size)
        self.gcn = GCN(hidden_dim, hidden_dim * 2, hidden_dim, dropout)
        self.encoderLayer = nn.TransformerEncoderLayer(hidden_dim, nhead=nhead, dim_feedforward=2 * hidden_dim,
                                                       batch_first=True)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tx, gx, mask=None):
        Tmask = ~mask

        encoder_output = self.encoderLayer(tx, src_key_padding_mask=Tmask) * mask.unsqueeze(-1)

        # 动态图学习
        edge_features, edge_weights = self.dynamic_graph(gx, mask)

        # 图卷积网络
        gcn_output = self.gcn(edge_features, edge_weights) * mask.unsqueeze(-1)

        # 计算加权边特征
        weighted_edge_features = torch.bmm(edge_weights, gcn_output)  # 结果形状 (batch_size, num_edges, feature_dim)

        combined_output = encoder_output + weighted_edge_features

        combined_output = self.norm(combined_output)

        return combined_output
