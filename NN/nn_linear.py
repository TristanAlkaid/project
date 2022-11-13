import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./data", train=False, download=True,transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64, drop_last=True)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output

tudui = Tudui()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    # output = torch.reshape(imgs, (1, 1, 1, -1)) batch_size, channel, height, width, -1为自己计算
    output = torch.flatten(imgs)
    print(output.shape)
    output = tudui.linear1(output)
    print(output)


