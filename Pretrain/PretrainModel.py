import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, edge_features: torch.Tensor, edge_weights: torch.Tensor):
        """
        :param edge_features: 形状为 (b, n, n, f)
        :param edge_weights: 形状为 (b, n, n, 1)
        :return: 输出特征，形状为 (b, n, out_features)
        """
        b, n, _, _ = edge_weights.shape

        # 扩展 edge_weights
        adj = edge_weights.expand(-1, -1, -1, edge_features.size(-1))  # 变为 (b, n, n, f)

        # 进行加权聚合
        aggregated_features = torch.matmul(adj, edge_features)  # (b, n, n, f) @ (b, n, n, f) -> (b, n, f)

        # 通过线性层进行变换
        output = self.linear(aggregated_features)
        return output

class normal_conv(nn.Module):
    def __init__(self, dg_hidden_size):
        super(normal_conv, self).__init__()
        self.fc1 = nn.Linear(dg_hidden_size, 1)
        self.fc2 = nn.Linear(dg_hidden_size * 2, dg_hidden_size)

    def forward(self, y, shape):
        support = self.fc1(torch.relu(self.fc2(y))).reshape(shape)
        return support

class DynamicGraphLearner(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super(DynamicGraphLearner, self).__init__()
        self.conv = normal_conv(hidden_dim)
        self.mlp_features = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)  # 输出维度调整为 feature_dim
        )
        self.mlp_weight = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出为单一值，用于权重
        )

    def forward(self, inputs: torch.Tensor):
        """
        :param inputs: 形状为 (b, n, f)
        :return: edge_features，形状为 (b, n, f)
        :return: edge_weights，形状为 (b, n, n)
        """
        batch_size, num_nodes, feature_dim = inputs.size()

        # 使用 self.conv 生成特征
        conv_output = self.conv(inputs)  # (b, n, f)

        # 计算边特征
        edge_features = self.mlp_features(conv_output.unsqueeze(2).expand(-1, -1, num_nodes, -1).view(-1, 2 * feature_dim))  # (b * n * n, f)

        # 计算边权重
        edge_weights = torch.sigmoid(self.mlp_weight(conv_output.unsqueeze(2).expand(-1, -1, num_nodes, -1).view(-1, 2 * feature_dim)))  # (b * n * n)

        # 将输出形状调整为 (b, n, f) 和 (b, n, n)
        edge_features = edge_features.view(batch_size, num_nodes, num_nodes, feature_dim)  # (b, n, n, f)
        edge_weights = edge_weights.view(batch_size, num_nodes, num_nodes)  # (b, n, n)

        return edge_features, edge_weights




class PretrainModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers):
        super(PretrainModel, self).__init__()
        self.embedding = nn.Linear(feature_dim, hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, nhead=8), num_layers=num_layers
        )
        self.dynamic_graph = DynamicGraphLearner(hidden_dim, hidden_dim * 2)  # GNN模块
        self.gcn_layer = GCNLayer(hidden_dim, hidden_dim)  # GCN层
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(hidden_dim, nhead=8), num_layers=num_layers
        )
        self.output_layer = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x, mask=None, is_pretraining=True):
        # 输入形状为 (batch, sequence_length, feature_dim)
        batch_size, seq_length, _ = x.size()
        x = self.embedding(x)  # 输入特征嵌入

        # 修改 mask 的维度
        if mask is not None:
            mask = mask.view(-1)  # 确保 mask 是 1-D

        # 使用 Transformer 编码器
        encoder_output = self.transformer_encoder(x.view(-1, x.size(2)), src_key_padding_mask=mask)  # 编码器输出

        # 将编码器输出传入 DynamicGraphModule
        edge_features, edge_weights = self.dynamic_graph(encoder_output.view(batch_size, seq_length, -1))

        # 使用 GCN 进行卷积
        gcn_output = self.gcn_layer(edge_features, edge_weights)

        # 将 GCN 输出与编码器输出相加
        combined_output = encoder_output.view(batch_size, seq_length, -1) + gcn_output  # 结合编码器输出和 GCN 输出

        if is_pretraining:
            # 使用解码器进行自回归填充
            decoder_output = self.transformer_decoder(combined_output.unsqueeze(0), combined_output.unsqueeze(0), tgt_key_padding_mask=mask.view(1, batch_size * seq_length))
            return self.output_layer(decoder_output)  # 输出层
        else:
            return encoder_output.view(batch_size, seq_length, -1) + gcn_output  # 下游任务输出


# 示例：创建模型
feature_dim = 10  # 输入特征维度
hidden_dim = 64  # 隐藏层维度
num_layers = 4  # Transformer层数
model = PretrainModel(feature_dim, hidden_dim, num_layers)

# 示例输入
input_data = torch.rand(32, 60, feature_dim)  # (batch size, sequence length, 特征维度)
mask = torch.zeros(32, 60, dtype=torch.bool)  # (batch size, sequence_length)

# 前向传播
output = model(input_data, mask, is_pretraining=True)
