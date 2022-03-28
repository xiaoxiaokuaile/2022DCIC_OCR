import pandas as pd
import numpy as np
import random
import torch
import os
from torch import nn
import time
from config.config import config
# loss及acc更新类
from utils import AverageMeter
# 加载数据用函数
from utils.data_loader import CaptDataset
from torch.utils.data import DataLoader
# 优化器
from torch.optim.lr_scheduler import CosineAnnealingLR
# loss
from utils.utils import *
# 学习率衰减策略
from torchtools.optim import Ranger,RAdam
# model
from model import Model_resnet18,Model_resnet101,EfficientNet_B4,EfficientNet_B5,EfficientNet_B0

torch.manual_seed(42)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 加载数据集
def dataloader(train_csv,val_csv,Isflod=False):
    if Isflod:
        train_csv = train_csv
        val_csv = val_csv
        num_train = len(train_csv)
        num_val = len(val_csv)
    else:
        train_fold = pd.read_csv('./data/font_all_aug_Kflod.csv')
        train_csv = train_fold[(train_fold['fold'] != 0)][['ImgName', 'label']]
        val_csv = train_fold[(train_fold['fold'] == 0) & (train_fold['aug'] == 'old')][['ImgName', 'label']]
        num_val = len(val_csv)
        num_train = len(train_csv)

    # 训练集训练
    trainDataset = CaptDataset(train_csv, config.img_path, data_mode='train')
    # 训练集验证
    trainDataset_val = CaptDataset(train_csv, config.img_path, data_mode='val')
    valDataset = CaptDataset(val_csv, config.img_path, data_mode='val')

    trainDataLoader = DataLoader(trainDataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    trainDataLoader_val = DataLoader(trainDataset_val, batch_size=config.batch_size, shuffle=False, num_workers=4)
    valDataLoader = DataLoader(valDataset, batch_size=config.batch_size*2, shuffle=False, num_workers=1)

    return trainDataLoader,trainDataLoader_val,valDataLoader,num_train, num_val


# 获取优化器及学习率衰减策略
def optimizer_scheduler(model,optim, lr):
    # 优化器
    optimizer_list = {'Ranger':Ranger(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr),}
    optimizer = optimizer_list[optim]
    # 学习率衰减策略
    scheduler_list = {'CosineLR_T1':CosineAnnealingLR(optimizer,config.epochs, eta_min=1e-6, last_epoch=-1),}
    scheduler = scheduler_list[lr]

    return optimizer, scheduler


def train_original(model):
    model.train()
    if torch.cuda.is_available():
        model = model.cuda()
    # 获取优化器, 学习率衰减策略
    optimizer,scheduler = optimizer_scheduler(model, 'Ranger', 'CosineLR_T1')
    # 加载数据
    trainDataLoader,trainDataLoader_val,valDataLoader,num_train, num_val = dataloader(train_csv=None, val_csv=None, Isflod=False)
    # 记录loss信息
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    best_acc = -1.
    # 训练Model
    for epoch in range(config.epochs):
        model.train()
        for circle, (x,label) in enumerate(trainDataLoader, 0):
            if torch.cuda.is_available():
                x = x.cuda()

            optimizer.zero_grad()
            output = model(x)
            # 计算损失
            loss = compute_loss(label, output)
            # 计算得分
            acc = calc_acc(label, output)

            loss_meter.add(loss.item())
            acc_meter.add(acc)
            loss.backward()
            optimizer.step()
        if True:
            # one epoch once
            scheduler.step()
            # 验证集准确率
            accuracy,test_loss = test(model, valDataLoader)
            # 保存最优模型
            if best_acc < accuracy:
                best_acc = accuracy
                # 保存最好的模型
                model.save("_best")
            model.save('_last')
            # -------------------------- 每个epoch输出结果 -----------------------------
            # 输出训练日志
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),end='')
            print("epoch:[%02d/%02d] | Learning rate: %.6f | Train loss %.5f | Train acc %.5f | Test loss %.5f | test acc %.3f | best acc %.3f" % \
                  (epoch,config.epochs, scheduler.get_last_lr()[0],loss_meter.avg,acc_meter.avg,test_loss, accuracy, best_acc))


# 验证集
def test(model, testDataLoader):
    model.eval()
    # 记录loss信息
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    for circle, (x, label) in enumerate(testDataLoader, 0):
        if torch.cuda.is_available():
            x = x.cuda()
        output = model(x)
        # 计算损失
        loss = compute_loss(label, output)
        # 计算得分
        acc = calc_acc(label, output)
        loss_meter.add(loss.item())
        acc_meter.add(acc)


    return  acc_meter.avg, loss_meter.avg


if __name__ == '__main__':
    net_list = {
                'EfficientNet_B0':EfficientNet_B0(),
                # 'EfficientNet_B1':EfficientNet_B1(),
                # 'EfficientNet_B2':EfficientNet_B2(),
                # 'EfficientNet_B3':EfficientNet_B3(),
                # 'EfficientNet_B4':EfficientNet_B4(),
                # 'EfficientNet_B5':EfficientNet_B5(),
                }
    net = net_list['EfficientNet_B0']
    # 加载预训练模型
    # net.load_model("./weights/senet_new.pth")
    # 交叉熵损失训练
    train_original(net)
