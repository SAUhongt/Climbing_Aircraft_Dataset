import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 输入的全连接层，将输入特征映射到 LSTM 隐藏层的维度
        self.input_fc = nn.Linear(input_size, hidden_size)

        # 定义 LSTM 层
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        # 输出的全连接层，用于将 LSTM 输出映射到目标输出维度
        self.output_fc = nn.Linear(hidden_size, input_size)

    def forward(self, x, input_mask):
        """
        x: 输入序列, shape = (batch_size, input_seq_length, input_size)
        input_mask: 输入序列掩码, shape = (batch_size, input_seq_length)
        output_mask: 输出序列掩码, shape = (batch_size, output_seq_length)
        """


        # 通过全连接层将输入特征映射到 LSTM 隐藏层维度
        x = self.input_fc(x)  # shape = (batch_size, input_seq_length, hidden_size)


        # LSTM 前向传播，得到 PackedSequence 输出
        lstm_out, _ = self.lstm(x)


        # 使用全连接层将 LSTM 输出映射到目标输出维度
        fc_output = self.output_fc(lstm_out)

        # 只取 output_mask 指定的长度
        output = fc_output * input_mask.unsqueeze(-1)

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
