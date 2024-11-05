import torch
from torch import nn


class DilatedInception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(DilatedInception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        cout = int(cout / len(self.kernel_set))

        for kern in self.kernel_set:
            # 计算填充以保持输出长度固定
            padding = (kern - 1) * dilation_factor // 2
            self.tconv.append(nn.Conv1d(cin, cout, kern, dilation=dilation_factor, padding=padding))

    def forward(self, input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))

        x = torch.cat(x, dim=1)  # 拼接通道
        return x
