import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

image_path = "./imgs/e850352ac65c10382a3013c1b2119313b07e8963.jpg"
image = Image.open(image_path)
print(image)
image = image.convert('RGB')

transfrom = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transfrom(image)
print(image.shape)


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
            Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),  # 根据官网公式计算 padding 和 stride
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(in_features=64 * 4 * 4, out_features=64),
            Linear(in_features=64, out_features=10),
        )

    def forward(self, x):
        x = self.model(x)
        return x


image = image.to(device)
model = torch.load("tudui_29_gpu.pth")
print(model)
image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)

print(output.argmax(1))
