from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):
    def __init__(self,root_dir, img_dir):
        self.root_dir = root_dir
        self.img_dir = img_dir
        self.path = os.path.join(self.root_dir, self.img_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        lable_dir = self.img_dir.split('_')[0] + '_lable'
        img_lable_path = os.path.join(self.root_dir, lable_dir)
        lable_arry = os.listdir(img_lable_path)
        img_lable_name = lable_arry[idx]
        lable_item_path = os.path.join(self.root_dir, lable_dir, img_lable_name)
        img_item_path = os.path.join(self.root_dir, self.img_dir, img_name)
        img = Image.open(img_item_path)
        with open(lable_item_path) as f:
            lable = f.read()
        return img,lable

    def __len__(self):
        return len(self.img_path)

root_dir = "dataset/train"
ants_img_dir = "ants_image"
bees_img_dir = "bees_image"
ants_dataset = MyData(root_dir, ants_img_dir)
bees_dataset = MyData(root_dir, bees_img_dir)

train_dataset = ants_dataset + bees_dataset