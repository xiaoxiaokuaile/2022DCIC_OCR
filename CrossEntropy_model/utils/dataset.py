import os
import numpy as np
import cv2
from PIL import Image
from torch.utils import data
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from config.parameters import *
import torch
import random
from .rec_img_aug import warp

nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
lower_char = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
upper_char = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']


# 将str标签转化为数值label
def StrtoLabel(Str):
    # print(Str)
    label = []
    for i in range(0, charNumber):
        if Str[i] >= '0' and Str[i] <= '9':
            # ord函数返回ASCII码值
            label.append(ord(Str[i]) - ord('0'))
        elif Str[i] >= 'a' and Str[i] <= 'z':
            label.append(ord(str(Str[i])) - ord("a") + 10)
        else:
            label.append(ord(str(Str[i])) - ord("A") + 36)
    return label


# 将label转化为标签
def LabeltoStr(Label):
    Str = ""
    for i in Label:
        if i <= 9:
            Str += chr(ord('0') + i)
        elif i <= 35:
            Str += chr(ord('a') + i - 10)
        else:
            Str += chr(ord('A') + i - 36)
    return Str


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
            # image = to_tensor(image)
            labelTensor = torch.Tensor(StrtoLabel(label))
            return image, labelTensor
        elif self.data_mode == 'val':
            # 验证集要有标签
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image = self.transform(image)
            # image = to_tensor(image)
            # 将标签转化为数值label
            labelTensor = torch.Tensor(StrtoLabel(label))
            return image, labelTensor
        elif self.data_mode == 'test':
            # 测试集正常处理
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image = self.transform(image)
            # image = to_tensor(image)
            return image, ImgName


