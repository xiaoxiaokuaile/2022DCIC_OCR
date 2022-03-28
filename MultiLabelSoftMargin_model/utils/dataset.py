import os
import cv2
from PIL import Image
from torch.utils import data
from torchvision import transforms
from config.parameters import *
import torch
from .rec_img_aug import warp

# 数据标签
source = [str(i) for i in range(0, 10)]
source += [chr(i) for i in range(97, 97 + 26)]
source += [chr(i) for i in range(65, 65 + 26)]
alphabet = ''.join(source)


# 将字符串转化为 248 大小的tensor
def StrtoTensor(label):
    target = []
    for char in label:
        vec = [0] * 62
        vec[alphabet.find(char)] = 1
        target += vec
    return target


# 加载数据集
class Captcha(data.Dataset):
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
            transforms.Resize((ImageHeight, ImageWidth)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.843, 0.846, 0.842], std=[0.196, 0.193, 0.198])
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
            # label
            labelTensor = torch.Tensor(StrtoTensor(label))
            return image, labelTensor
        elif self.data_mode == 'val':
            # 验证集要有标签
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image = self.transform(image)
            # 将标签转化为数值label
            labelTensor = torch.Tensor(StrtoTensor(label))
            return image, labelTensor
        elif self.data_mode == 'test':
            # 测试集正常处理
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image = self.transform(image)
            return image, ImgName


