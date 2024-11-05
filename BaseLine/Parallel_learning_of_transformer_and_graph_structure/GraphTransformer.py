import torch
import torch.nn as nn

from BaseLine.Parallel_learning_of_transformer_and_graph_structure.Dilated_inception_Layer.Dilated_inception import \
    DilatedInception
from BaseLine.Parallel_learning_of_transformer_and_graph_structure.TransformerEncoder_graph.CustomTransformerEncoder import \
    CustomTransformerEncoder
from Pretrain.PretrainModel import PatchEmbedding, PositionalEncoding, DynamicGraphLearner, GCN


class GraphTransformerBaseline(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers=1, dropout=0.1, seq_len=60, patch_size=1):
        super(GraphTransformerBaseline, self).__init__()
        self.hidden_size = hidden_dim
        self.num_layers = num_layers

        self.patch_embedding = PatchEmbedding(feature_dim, hidden_dim, patch_size)
        self.positional_encoding = PositionalEncoding(hidden_dim, seq_len)
        self.dilated_inception = DilatedInception(cin=feature_dim, cout=hidden_dim)
        self.layers = nn.ModuleList([
            CustomTransformerEncoder(hidden_dim, nhead=8, dropout=dropout, seq_len=seq_len,
                                     patch_size=patch_size)
            for _ in range(num_layers)
        ])
        self.linear = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x, input_mask, output_mask):
        _, target_len = output_mask.size()

        tx = self.patch_embedding(x)
        tx = self.positional_encoding(tx)
        gx = self.dilated_inception(x.permute(0, 2, 1)).permute(0, 2, 1) * input_mask.unsqueeze(-1)

        layer_output = None

        for index, layer in enumerate(self.layers):
            if index == 0:
                layer_output = layer(tx, gx, input_mask) * input_mask.unsqueeze(-1)
            else:
                layer_output = layer(layer_output, layer_output, input_mask) * input_mask.unsqueeze(-1)

        output = self.linear(layer_output)

        # 只取 output_mask 指定的长度
        output = output[:, :target_len, :] * output_mask.unsqueeze(-1)

        return output
