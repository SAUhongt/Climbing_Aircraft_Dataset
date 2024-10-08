import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv  # 从PyTorch几何库中导入图卷积网络层（GCNConv）
from torch_geometric.utils import dense_to_sparse, to_dense_batch, to_dense_adj
from torch_geometric.data import Data, Batch


class GCN(torch.nn.Module):  # 定义一个GNN类，继承自PyTorch的Module类
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):  # 定义GNN类的初始化函数
        super().__init__()  # 调用父类（Module类）的初始化函数
        # 创建第一个图卷积层，输入特征维度为数据集节点特征维度，输出特征维度为16
        self.conv1 = GCNConv(input_dim, hidden_dim)
        # 创建第二个图卷积层，输入特征维度为16，输出特征维度为数据集类别数量
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout  # 定义Dropout层，用于防止过拟合

    def forward(self, edge_features, edge_weights):  # 定义前向传播函数，接受一个数据对象作为输入

        batch = self.create_graph_batch(edge_features, edge_weights)

        x = self.conv1(batch.x, batch.edge_index, batch.edge_attr)  # 通过第一个图卷积层处理节点特征
        x = F.relu(x)  # 对输出进行ReLU激活函数操作
        x = F.dropout(x, self.dropout, training=self.training)  # 对输出进行Dropout操作，用于防止过拟合
        x = self.conv2(x, batch.edge_index, batch.edge_attr)  # 通过第二个图卷积层处理节点特征
        x = F.relu(x)  # 对输出进行ReLU激活函数操作
        out, _ = to_dense_batch(x, batch.batch)
        return out

    @staticmethod
    def create_graph_batch(features: torch.Tensor, adj_matrices: torch.Tensor) -> Batch:
        """
        将特征矩阵和邻接矩阵转换为 PyTorch Geometric 格式，并打包为 Batch 对象。

        参数:
        - features: torch.Tensor, 形状为 (batch_size, num_nodes, feature_dim)，表示每个图的节点特征。
        - adj_matrices: torch.Tensor, 形状为 (batch_size, num_nodes, num_nodes)，表示每个图的邻接矩阵。

        返回:
        - Batch 对象，包含所有图数据的批次。
        """
        data_list = []
        batch_size = features.size(0)

        for i in range(batch_size):
            # 获取当前图的特征矩阵和邻接矩阵
            feature_matrix = features[i]  # (num_nodes, feature_dim)
            adj_matrix = adj_matrices[i]  # (num_nodes, num_nodes)

            # 将邻接矩阵转换为 edge_index 格式
            edge_index, edge_weight = dense_to_sparse(adj_matrix)  # edge_index: (2, E), edge_weight: (E,)

            # 创建图数据对象
            data = Data(x=feature_matrix, edge_index=edge_index, edge_attr=edge_weight)
            data_list.append(data)

        # 使用 Batch 将图数据打包
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
        self.linear = nn.Linear(patch_size * input_dim, embed_dim)  # 修改为适应展平后的输入

    def forward(self, x):
        batch_size, seq_length, input_dim = x.size()
        num_patches = seq_length // self.patch_size
        x = x.view(batch_size, num_patches,
                   self.patch_size * input_dim)  # (batch_size, num_patches, patch_size * input_dim)
        return self.linear(x)  # (batch_size, num_patches, embed_dim)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.encoding = self._generate_positional_encoding(embed_dim, max_len)

    def _generate_positional_encoding(self, embed_dim, max_len):
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        idx = torch.arange(embed_dim, dtype=torch.float).unsqueeze(0)  # 这里修复了idx的定义
        encoding = pos / 10000 ** (idx / embed_dim)  # Shape: (max_len, embed_dim)
        encoding[:, 0::2] = torch.sin(encoding[:, 0::2])
        encoding[:, 1::2] = torch.cos(encoding[:, 1::2])
        return encoding.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        # x: (batch_size, seq_length, embed_dim)
        return x + self.encoding[:, :x.size(1)]  # Broadcast positional encoding


class PretrainModel(nn.Module):
    def __init__(self, feature_dim, seq_len, hidden_dim, num_layers, dropout, patch_size):
        super(PretrainModel, self).__init__()
        self.patch_embedding = PatchEmbedding(feature_dim, hidden_dim, patch_size)
        self.positional_encoding = PositionalEncoding(hidden_dim, seq_len)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, nhead=8, batch_first=True), num_layers=num_layers
        )
        self.dynamic_graph = DynamicGraphLearner(hidden_dim, hidden_dim * 2, hidden_dim, seq_len // patch_size)
        self.gcn = GCN(hidden_dim, hidden_dim * 2, hidden_dim, dropout)  # 使用新的 GCN
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, nhead=8, batch_first=True), num_layers=num_layers
        )
        self.output_layer = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x, mask=None, is_pretraining=True):
        # batch_size, seq_length, feature_dim = x.size()
        x = self.patch_embedding(x)
        x = self.positional_encoding(x)
        batch_size, seq_length, feature_dim = x.size()

        # 编码器
        encoder_output = self.encoder(x, src_key_padding_mask=mask.view(batch_size,
                                                                                    -1) if mask is not None else None)

        # 将编码器输出传入 DynamicGraphModule
        edge_features, edge_weights = self.dynamic_graph(encoder_output)

        # 使用 GCN 进行卷积
        gcn_output = self.gcn(edge_features, edge_weights)  # 调整输入格式

        # 将 GCN 输出与编码器输出相加
        combined_output = encoder_output + gcn_output.view(batch_size, seq_length, -1)  # 调整输出格式

        if is_pretraining:

            # 使用解码器进行自回归填充
            decoder_output = self.decoder(
                combined_output,
                src_key_padding_mask=mask.view(batch_size,
                                               -1) if mask is not None else None
            )
            return self.output_layer(decoder_output.view(batch_size*seq_length, -1))
        else:
            return combined_output


# 示例参数
feature_dim = 10  # 输入特征维度
hidden_dim = 64  # 隐藏层维度
num_layers = 4  # Transformer层数
seq_len = 60
patch_size = 5  # 根据需要设置
dropout = 0.5  # dropout比例

# 创建模型
model = PretrainModel(feature_dim, seq_len, hidden_dim, num_layers, dropout, patch_size)

# 示例输入
input_data = torch.rand(32, 60, feature_dim)  # (batch size, sequence length, 特征维度)
mask = torch.zeros(32, 60 // patch_size, dtype=torch.bool)  # (batch size, num_patches)

# 前向传播
output = model(input_data, mask, is_pretraining=True)
