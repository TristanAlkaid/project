import torch
import torchvision
from PIL import Image

from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

image_path = "./imgs/cat.png"
image = Image.open(image_path)
print(image)
image = image.convert('RGB')

transfrom = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transfrom(image)
print(image.shape)

image = image.to(device)
model = torch.load("alex_net_50.pth")
image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output, "\n")

category_arry = ['airplane', "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
id = output.argmax(1)
print("输入的图片内容是：{}".format(category_arry[id]))
