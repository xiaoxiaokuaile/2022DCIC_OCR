import pandas as pd
import numpy as np
import random
import torch
from torch import nn
import time
from config.parameters import *
# loss及acc更新类
from utils import AverageMeter,calculat_acc
# 加载数据用函数
from utils.dataset import *
from torch.utils.data import DataLoader
# 优化器
from torch.optim import Adam,SGD,RMSprop
from utils.optimizer import RAdam, Ranger,AdamW, LabelSmoothSoftmaxCE, LSR,AdaBound,AdaBoundW
# loss
from torch.nn import MultiLabelSoftMarginLoss
# 学习率衰减策略
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts,CosineAnnealingLR,ReduceLROnPlateau,StepLR,MultiStepLR
from utils.lr_scheduler.scheduler import GradualWarmupScheduler
# model
from model import EfficientNet_B0,EfficientNet_B1,EfficientNet_B2,EfficientNet_B3,EfficientNet_B4
from model import EfficientNet_B5,EfficientNet_B6,EfficientNet_B7,EfficientNet_B8
from model import RepVGG_A0,RepVGG_A1,RepVGG_A2, RepVGG_B0,RepVGG_B1,RepVGG_B1g2,RepVGG_B1g4,RepVGG_B2
from model import RepVGG_B2g4,RepVGG_B3,RepVGG_B3g4,RepVGG_D2se

torch.manual_seed(42)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 加载数据集
def dataloader():
    train_fold = pd.read_csv('./data/font_all_aug_Kflod.csv')
    train_csv = train_fold[(train_fold['fold'] != 0) & (train_fold['aug'] == 'old')][['ImgName', 'label']]
    # & (train_fold['font'] == 3)
    val_csv = train_fold[(train_fold['fold'] == 0) & (train_fold['aug'] == 'old')][['ImgName', 'label']]

    # 训练集训练
    trainDataset = Captcha(train_csv, img_path, data_mode='train')
    trainDataLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, num_workers=4)
    # 验证集
    valDataset = Captcha(val_csv, img_path, data_mode='val')
    valDataLoader = DataLoader(valDataset, batch_size=2, shuffle=False, num_workers=1)

    return trainDataLoader,valDataLoader


# 获取优化器及学习率衰减策略
def optimizer_scheduler(model,optim, lr):
    # 优化器
    optimizer_list = {'SGD':SGD(model.parameters(), lr=learningRate),
                      'Adam':Adam(model.parameters(), lr=learningRate),
                      'RAdam':RAdam(model.parameters(), lr=learningRate,betas=(0.9, 0.999),weight_decay=6.5e-4),
                      'Ranger':Ranger(filter(lambda p: p.requires_grad, model.parameters()), lr=learningRate),
                      # 训练速度Adam一样块,SGD效果一样好
                      'AdaBound':AdaBound(model.parameters(), lr=learningRate, final_lr=0.1, gamma=1e-4)}
    optimizer = optimizer_list[optim]
    # 学习率衰减策略
    scheduler_after = StepLR(optimizer, step_size=20, gamma=0.5)
    milestone_list = [10 * k for k in range(1, totalEpoch // 10)]
    scheduler_list = {
                      # T_max=5 余弦学习率周期性变化5次
                      'CosineAnnealingLR':CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6), # Cosine需要的初始lr比较大1e-2,1e-3都可以
                      # 周期为1的余弦衰减策略
                      'CosineLR_T1':CosineAnnealingLR(optimizer,totalEpoch, eta_min=1e-6, last_epoch=-1),
                      'ReduceLROnPlateau':ReduceLROnPlateau(optimizer, 'min', patience=4),
                      'GradualWarmupScheduler':GradualWarmupScheduler(optimizer,8,10,after_scheduler=scheduler_after),
                      'MultiStepLR':MultiStepLR(optimizer, milestones=milestone_list, gamma=0.5),  # lr 3e-3 best
                      'StepLR':StepLR(optimizer, step_size=20, gamma=0.5)}
    scheduler = scheduler_list[lr]

    return optimizer, scheduler


def train_original(model):
    model.train()
    if torch.cuda.is_available():
        model = model.cuda()
    # loss 损失函数
    criterion = MultiLabelSoftMarginLoss()
    # 获取优化器, 学习率衰减策略
    optimizer,scheduler = optimizer_scheduler(model, 'RAdam', 'CosineLR_T1')
    # 加载数据
    trainDataLoader,valDataLoader = dataloader()

    best_acc = -1.
    # 训练Model
    for epoch in range(totalEpoch):
        model.train()
        # 记录loss信息
        loss_meter = AverageMeter()
        train_acc = AverageMeter()
        for circle, (x,label) in enumerate(trainDataLoader, 0):
            if torch.cuda.is_available():
                x = x.cuda()
                label = label.cuda()

            # 使用MixUp数据增强
            if alpha > 0 and random.random() < mixup_prob:
                lam_aa = np.random.beta(alpha,alpha)
                # 随机打乱索引顺序
                rand_index = torch.randperm(x.size()[0])
                if torch.cuda.is_available():
                    rand_index =rand_index.cuda()
                # 获取 label
                label_aa = label
                label_bb = label[rand_index]
                # mix图片
                x[:, :, :, :] = x[:,:,:,:] * lam_aa + x[rand_index, :, :, :] * (1. - lam_aa)  # (bz,3,128,320)
                optimizer.zero_grad()
                output = model(x)
                # 计算loss
                loss = criterion(output,label_aa) * lam_aa + criterion(output,label_bb) * (1. - lam_aa)
            else:
                optimizer.zero_grad()
                output = model(x)
                # 计算损失
                loss= criterion(output, label)
                # 计算训练集准确率
                acc = calculat_acc(output, label)
                train_acc.add(float(acc))

            loss_meter.add(loss.item())
            loss.backward()
            optimizer.step()
        if True:
            # one epoch once
            scheduler.step()
            # ------------------ 验证集准确率 -------------------
            model.eval()
            # 记录loss信息
            test_loss = AverageMeter()
            test_acc = AverageMeter()
            for circle, (x, label) in enumerate(valDataLoader, 0):
                if torch.cuda.is_available():
                    x = x.cuda()
                    label = label.cuda()
                output = model(x)
                # 计算损失
                loss= criterion(output, label)
                test_loss.add(loss.item())
                # 计算准确率
                t_acc = calculat_acc(output, label)
                test_acc.add(float(t_acc))
            # -------------------------------------------------
            # 保存最优模型
            if best_acc < test_acc.avg:
                best_acc = test_acc.avg
                # 保存最好的模型
                model.save("_best")
            model.save('_last')
            # -------------------------- 每个epoch输出结果 -----------------------------
            # 输出训练日志
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),end='')
            print("epoch:[%02d/%02d] | Learning rate: %.6f | Train loss %.5f | Train acc %.5f| Test loss %.5f | test acc %.3f | best acc %.3f" % \
                  (epoch,totalEpoch, scheduler.get_last_lr()[0],loss_meter.avg,train_acc.avg,test_loss.avg,test_acc.avg ,best_acc))


if __name__ == '__main__':
    net_list = {
                'EfficientNet_B0':EfficientNet_B0(),
                # 'EfficientNet_B1':EfficientNet_B1(),
                # 'EfficientNet_B2':EfficientNet_B2(),
                # 'EfficientNet_B3':EfficientNet_B3(),
                # 'EfficientNet_B4':EfficientNet_B4(),
                # 'EfficientNet_B5':EfficientNet_B5(),
                # 'EfficientNet_B6':EfficientNet_B6(),
                # 'EfficientNet_B7':EfficientNet_B7(),
                # 'EfficientNet_B8':EfficientNet_B8(),
                # 'EfficientNet_B0_CRNN': EfficientNet_B0_CRNN(),
                # 'EfficientNet_B0_sig': EfficientNet_B0_sig(),
                # 'RepVGG_A0':RepVGG_A0(),
                # 'RepVGG_A1':RepVGG_A1(),
                # 'RepVGG_A2':RepVGG_A2(),
                # 'RepVGG_B0':RepVGG_B0(),
                # 'RepVGG_B1':RepVGG_B1(),
                # 'RepVGG_B1g2':RepVGG_B1g2(),
                # 'RepVGG_B1g4':RepVGG_B1g4(),
                # 'RepVGG_B2':RepVGG_B2(),
                # 'RepVGG_B2g4':RepVGG_B2g4(),
                # 'RepVGG_B3':RepVGG_B3(),
                # 'RepVGG_B3g4':RepVGG_B3g4(),
                # 'RepVGG_D2se':RepVGG_D2se(),   # 图像分辨率要足够高才可
                }
    net = net_list['EfficientNet_B0']
    # 加载预训练模型
    # net.load_model("./weights/senet_new.pth")
    # 交叉熵损失训练
    train_original(net)
