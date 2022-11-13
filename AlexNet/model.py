import torch
from torch import nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # 定义每一个层
        self.layer1 = nn.Sequential(
            # 卷积层
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 根据官网公式计算 padding 和 stride
            # 池化层
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 激活函数 Relu
            nn.ReLU(),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(),
        )

        # 定义全连接层
        self.fc1 = nn.Linear(256 * 3 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(-1, 256 * 3 * 3)
        x = self.fc1(x)
        x = F.dropout(x, p=0.5)
        x = self.fc2(x)
        x = F.dropout(x, p=0.5)
        x = self.fc3(x)

        return x


if __name__ == '__main__':
    alex_net = AlexNet()
    input = torch.ones((64, 3, 224, 224))
    output = alex_net(input)
    print(output.shape)