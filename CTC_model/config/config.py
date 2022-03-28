# Learner: 王振强
# Learn Time: 2022/2/3 23:26
import torch.nn as nn
# 针对数据增强专门写的API
import albumentations as alb
import cv2
import torch


class config:
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    pre_weight = None
    criterion = nn.CTCLoss(blank=0)

    source = [str(i) for i in range(0, 10)]
    source += [chr(i) for i in range(97, 97 + 26)]
    source += [chr(i) for i in range(65, 65 + 26)]
    alphabet = ''.join(source)
    letters = sorted(list(set(list(alphabet))))
    # 字符list
    vocabulary = ["-"] + letters
    # 索引字典
    idx2char = {k: v for k, v in enumerate(vocabulary, start=0)}
    char2idx = {v: k for k, v in idx2char.items()}
    H, W = 128, 320  # (128,256)->(4,8)   (128,320)->(4,10)

    # 带上 - 63
    num_chars = len(char2idx)

    batch_size = 32
    lr = 1e-3
    epochs = 60
    weight_decay = 1e-4

    # 数据集路径
    img_path = "./data/font_all_aug_train"
    # 测试集图片地址
    testPath = "./data/test"



