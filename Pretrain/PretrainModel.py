import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse
from torch_geometric.utils import to_dense_batch


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, device):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout
        self.device = device

    def forward(self, edge_features, edge_weights):
        batch = self.create_graph_batch(edge_features, edge_weights).to(self.device)
        x = self.conv1(batch.x, batch.edge_index, batch.edge_attr)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, batch.edge_index, batch.edge_attr)
        x = F.relu(x)
        out, _ = to_dense_batch(x, batch.batch)
        return out

    @staticmethod
    def create_graph_batch(features: torch.Tensor, adj_matrices: torch.Tensor) -> Batch:
        data_list = []
        batch_size = features.size(0)

        for i in range(batch_size):
            feature_matrix = features[i]
            adj_matrix = adj_matrices[i]
            edge_index, edge_weight = dense_to_sparse(adj_matrix)
            data = Data(x=feature_matrix, edge_index=edge_index, edge_attr=edge_weight)
            data_list.append(data)

        batch = Batch.from_data_list(data_list)
        return batch


class NormalConv(nn.Module):
    def __init__(self, dg_hidden_size):
        super(NormalConv, self).__init__()
        self.fc1 = nn.Linear(dg_hidden_size * 2, dg_hidden_size * 2)
        self.fc2 = nn.Linear(dg_hidden_size, dg_hidden_size * 2)

    def forward(self, y, shape):
        support = self.fc1(torch.relu(self.fc2(y))).reshape(shape)
        return support


class DynamicGraphLearner(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, out_feature_dim: int, seq_len: int):
        super(DynamicGraphLearner, self).__init__()
        self.conv = NormalConv(feature_dim)
        self.mlp_features = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_feature_dim)
        )
        self.mlp_weight = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len),
            nn.Tanh()
        )

    def forward(self, inputs: torch.Tensor):
        batch_size, num_nodes, feature_dim = inputs.size()
        conv_output = self.conv(inputs.view(-1, feature_dim), (-1, feature_dim * 2))
        edge_features = self.mlp_features(conv_output)
        edge_weights = torch.relu(self.mlp_weight(conv_output))

        edge_features = edge_features.view(batch_size, num_nodes, feature_dim)
        edge_weights = edge_weights.view(batch_size, num_nodes, num_nodes)

        return edge_features, edge_weights


class PatchEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim, patch_size):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.linear = nn.Linear(patch_size * input_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_length, input_dim = x.size()
        num_patches = seq_length // self.patch_size
        x = x.view(batch_size, num_patches, self.patch_size * input_dim)
        return self.linear(x)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.encoding = self._generate_positional_encoding(embed_dim, max_len)

    def _generate_positional_encoding(self, embed_dim, max_len):
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        idx = torch.arange(embed_dim, dtype=torch.float).unsqueeze(0)
        encoding = pos / 10000 ** (idx / embed_dim)
        encoding[:, 0::2] = torch.sin(encoding[:, 0::2])
        encoding[:, 1::2] = torch.cos(encoding[:, 1::2])
        return encoding.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        # x: (batch_size, seq_length, embed_dim)
        encoding = self.encoding[:, :x.size(1)].to(x.device)  # 将 self.encoding 转移到与 x 相同的设备
        return x + encoding


class PretrainModel(nn.Module):
    def __init__(self, feature_dim, seq_len, hidden_dim, num_layers, dropout, patch_size, device):
        super(PretrainModel, self).__init__()
        self.device = device
        self.patch_embedding = PatchEmbedding(feature_dim, hidden_dim, patch_size).to(device)
        self.positional_encoding = PositionalEncoding(hidden_dim, seq_len).to(device)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, nhead=8, batch_first=True), num_layers=num_layers
        ).to(device)
        self.dynamic_graph = DynamicGraphLearner(hidden_dim, hidden_dim * 2, hidden_dim, seq_len // patch_size).to(device)
        self.gcn = GCN(hidden_dim, hidden_dim * 2, hidden_dim, dropout, device).to(device)
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, nhead=8, batch_first=True), num_layers=num_layers
        ).to(device)
        self.output_layer = nn.Linear(hidden_dim, feature_dim).to(device)

    def forward(self, x, mask=None, is_pretraining=True):
        x = x.to(self.device)
        mask = mask.to(self.device) if mask is not None else None

        x = self.patch_embedding(x)
        x = self.positional_encoding(x)
        batch_size, seq_length, feature_dim = x.size()

        encoder_output = self.encoder(
            x, src_key_padding_mask=mask.view(batch_size, -1) if mask is not None else None
        )

        edge_features, edge_weights = self.dynamic_graph(encoder_output)

        gcn_output = self.gcn(edge_features, edge_weights)
        combined_output = encoder_output + gcn_output.view(batch_size, seq_length, -1)

        if is_pretraining:
            decoder_output = self.decoder(
                combined_output, src_key_padding_mask=mask.view(batch_size, -1) if mask is not None else None
            )
            return self.output_layer(decoder_output.view(batch_size * seq_length, -1))
        else:
            return combined_output




#调试维度用的一个小文件
# pretraining_data_path = 'E:\\climbing-aircraft-dataset\\dataTest\\pretraining_data.pt'  # 指定保存的文件名
# pretraining_data_valid_path = 'E:\\climbing-aircraft-dataset\\dataTest\\pretraining_data_valid.pt'  # 指定保存的文件名


# pretraining_data_path = 'E:\\climbing-aircraft-dataset\\pretraining_data\\pretraining_data.pt'
# pretraining_data_valid_path = 'E:\\climbing-aircraft-dataset\\pretraining_data\\pretraining_data_valid.pt'


# 仅计算被抹除部分的损失
    # l1 = nn.MSELoss()(outputs * Nfea_masks * valid_mask_expanded, sequences * Nfea_masks * valid_mask_expanded)
    # 计算所有有效区域的损失
    # l2 = nn.MSELoss()(outputs * valid_mask_expanded, sequences * valid_mask_expanded)