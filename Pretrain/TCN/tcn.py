import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        # 定义裁剪的大小
        self.chomp_size = chomp_size

    def forward(self, x):
        # 裁剪掉输入数据最后 chomp_size 个元素
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # 定义第一个卷积层，使用 weight_norm 进行权重归一化
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 定义第一个裁剪层，裁剪掉卷积后多余的 padding
        self.chomp1 = Chomp1d(padding)
        # 定义第一个激活函数
        self.relu1 = nn.ReLU()
        # 定义第一个 dropout 层
        self.dropout1 = nn.Dropout(dropout)

        # 定义第二个卷积层，使用 weight_norm 进行权重归一化
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 定义第二个裁剪层，裁剪掉卷积后多余的 padding
        self.chomp2 = Chomp1d(padding)
        # 定义第二个激活函数
        self.relu2 = nn.ReLU()
        # 定义第二个 dropout 层
        self.dropout2 = nn.Dropout(dropout)

        # 将上述层组合成一个序列
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        # 如果输入和输出的通道数不同，使用 1x1 卷积进行降维
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs!= n_outputs else None
        # 定义最终的激活函数
        self.relu = nn.ReLU()
        # 初始化权重
        self.init_weights()

    def init_weights(self):
        # 初始化卷积层权重
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        # 如果存在降维卷积，初始化其权重
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # 前向传播，计算网络的输出
        out = self.net(x)
        # 如果存在降维卷积，计算其输出
        res = x if self.downsample is None else self.downsample(x)
        # 将网络输出和降维卷积输出相加，并通过激活函数
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        # 定义网络的层列表
        layers = []
        # 获取网络层数
        num_levels = len(num_channels)
        # 遍历每一层
        for i in range(num_levels):
            # 计算 dilation size
            dilation_size = 2 ** i
            # 计算输入通道数
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            # 计算输出通道数
            out_channels = num_channels[i]
            # 添加一个 TemporalBlock 到网络中
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        # 将所有层组合成一个序列
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # 前向传播，计算网络的输出
        return self.network(x)
