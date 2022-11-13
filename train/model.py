import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.module = nn.Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),  # 根据官网公式计算 padding 和 stride
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(in_features=64*4*4, out_features=64),
            Linear(in_features=64, out_features=10),
        )

    def forward(self, x):
        x = self.module(x)
        return x


if __name__ == '__main__':
    tudui = Tudui()
    input = torch.ones((64, 3, 32, 32))
    output = tudui(input)
    print(output.shape)