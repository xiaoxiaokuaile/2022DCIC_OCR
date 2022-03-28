# Learner: 王振强
# Learn Time: 2022/2/3 23:28
from torch.utils.data import Dataset
import pandas as pd
import cv2
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from config.config import config
import os
from .rec_img_aug import warp
from PIL import Image


# 加载数据集
class CaptDataset(Dataset):
    # 输入已经读取的CSV表格, 图片地址, 是否测试集
    def __init__(self, csv, img_path, data_mode='train'):
        # 表格数据
        self.data = csv
        # 图片地址
        self.img_path = img_path
        # 是否是测试集
        self.data_mode = data_mode
        # 图像处理
        self.transform = transforms.Compose([
            transforms.Resize((config.H, config.W)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 获取图片名称及label
        ImgName, label = self.data.iloc[index, :]
        # 获取某一张图片地址
        imgPath = os.path.join(self.img_path, ImgName)
        # 打开图片 (有的图片4通道)
        # image = Image.open(imgPath).convert('RGB')
        image = cv2.cvtColor(cv2.imread(imgPath), cv2.COLOR_BGR2RGB)

        if self.data_mode == 'train':
            # 训练集要做数据增强
            image = warp(image)
            # resize + Norm归一化
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image = self.transform(image)
            return image, label

        elif self.data_mode == 'val':
            # 验证集要有标签
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image = self.transform(image)
            return image, label

        elif self.data_mode == 'test':
            # 测试集正常处理
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image = self.transform(image)
            return image, ImgName