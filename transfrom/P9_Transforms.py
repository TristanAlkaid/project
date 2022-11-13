from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import cv2

# python的用法 -> tensor数据类型
# 通过 transforms.Totensor去看两个问题

# 2. 为什么需要 Tensor 数据类型

# 绝对路径： C:\Users\Trist\Desktop\campus\pythonProject\data\train\ants_image\0013035.jpg
# 相对路径： data/train/ants_image/0013035.jpg
img_path = "data/train/ants_image/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")

# 1. transforms 如何使用（python）
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img", tensor_img)

writer.close()