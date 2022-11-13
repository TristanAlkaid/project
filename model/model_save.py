import torch
import torchvision
from torch import nn
from torch.nn import Sequential, MaxPool2d, Flatten, Linear
from torch.nn.qat import Conv2d
from nn_seq import *



# vgg16 = torchvision.models.vgg16(pretrained=False)
# 保存方式 1 （模型结构 + 模型参数），加载模型
# torch.save(vgg16, "vgg16_method1.pth")
# model1 = torch.load("vgg16_method1.pth")

# 保存方式 2 （模型参数）， 加载模型
# torch.save(vgg16.state_dict(), "vgg16_method2.pth")
# model2 = torch.load("vgg16_method2.pth")
# vgg16 = torchvision.models.vgg16(pretrained=False)
# vgg16.state_dict(torch.load("vgg16_method2.pth"))

# class Tudui(nn.Module):
#     def __init__(self):
#         super(Tudui, self).__init__()
#         self.model1 = Sequential(
#             Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),  # 根据官网公式计算 padding 和 stride
#             MaxPool2d(kernel_size=2),
#             Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding=2),
#             MaxPool2d(kernel_size=2),
#             Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
#             MaxPool2d(kernel_size=2),
#             Flatten(),
#             Linear(in_features=1024, out_features=64),
#             Linear(in_features=64, out_features=10),
#         )
#
#     def forward(self, x):
#         x = self.model1(x)
#         return x


# torch.save(Tudui, "Tudui_method1.pth")

model1 = torch.load("Tudui_method1.pth")