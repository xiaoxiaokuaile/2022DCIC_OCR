import pandas as pd
import numpy as np
import random
import torch
from torch import nn
import time
from config.parameters import *
# loss及acc更新类
from utils import AverageMeter
# 加载数据用函数
from utils.dataset import *
from torch.utils.data import DataLoader
# 优化器
from torch.optim import Adam,SGD,RMSprop
from utils.optimizer import RAdam, Ranger,AdamW, LabelSmoothSoftmaxCE, LSR,AdaBound,AdaBoundW
# loss
from torch.nn import CrossEntropyLoss
# 学习率衰减策略
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts,CosineAnnealingLR,ReduceLROnPlateau,StepLR,MultiStepLR
from utils.scheduler import GradualWarmupScheduler
# model
from model import EfficientNet_B0,EfficientNet_B1,EfficientNet_B2,EfficientNet_B3,EfficientNet_B4
from model import EfficientNet_B5,EfficientNet_B6,EfficientNet_B7,EfficientNet_B8
from model import EfficientNet_B0_CRNN,EfficientNet_B0_sig
from model import RepVGG_A0,RepVGG_A1,RepVGG_A2, RepVGG_B0,RepVGG_B1,RepVGG_B1g2,RepVGG_B1g4,RepVGG_B2
from model import RepVGG_B2g4,RepVGG_B3,RepVGG_B3g4,RepVGG_D2se

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
        train_fold = pd.read_csv('./data/train_flod.csv')
        train_csv = train_fold[train_fold['fold'] != 0][['ImgName', 'label']]
        val_csv = train_fold[train_fold['fold'] == 0][['ImgName', 'label']]
        num_val = len(val_csv)
        num_train = len(train_csv)

    # 训练集训练
    trainDataset = Captcha(train_csv, img_path, data_mode='train')
    # 训练集验证
    trainDataset_val = Captcha(train_csv, img_path, data_mode='val')
    valDataset = Captcha(val_csv, img_path, data_mode='val')

    trainDataLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, num_workers=4)
    trainDataLoader_val = DataLoader(trainDataset_val, batch_size=batchSize, shuffle=False, num_workers=4)
    valDataLoader = DataLoader(valDataset, batch_size=2, shuffle=False, num_workers=1)

    return trainDataLoader,trainDataLoader_val,valDataLoader,num_train, num_val


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


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # 获取切割中心
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bbx1 = 0
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    # bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bbx2 = W
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


# 标签平滑 LSR
class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def train_original(model):
    model.train()
    if torch.cuda.is_available():
        model = model.cuda()
    # loss 损失函数
    criterion = CrossEntropyLoss()
    # criterion = LabelSmoothing()
    # 获取优化器, 学习率衰减策略
    optimizer,scheduler = optimizer_scheduler(model, 'RAdam', 'CosineLR_T1')
    # 加载数据
    trainDataLoader,trainDataLoader_val,valDataLoader,num_train, num_val = dataloader(train_csv=None, val_csv=None, Isflod=False)
    # 记录loss信息
    loss_meter = AverageMeter()
    best_acc = -1.
    # 训练Model
    for epoch in range(totalEpoch):
        model.train()
        for circle, (x,label) in enumerate(trainDataLoader, 0):
            if torch.cuda.is_available():
                x = x.cuda()
                label = label.cuda()
            label = label.long()  # (batchsize,4)
            r = random.random()
            # 使用MixUp数据增强
            if alpha > 0 and r < mixup_prob:
                lam_aa = np.random.beta(alpha,alpha)
                # 随机打乱索引顺序
                rand_index = torch.randperm(x.size()[0])
                if torch.cuda.is_available():
                    rand_index =rand_index.cuda()
                # 获取 label
                label_aa = label
                label1_aa, label2_aa, label3_aa, label4_aa = label_aa[:, 0], label_aa[:, 1], label_aa[:, 2], label_aa[:, 3]
                label_bb = label[rand_index]
                label1_bb, label2_bb, label3_bb, label4_bb = label_bb[:, 0], label_bb[:, 1], label_bb[:, 2], label_bb[:, 3]  # (batchsize)
                # mix图片
                x[:, :, :, :] = x[:,:,:,:] * lam_aa + x[rand_index, :, :, :] * (1. - lam_aa)  # (bz,3,128,320)
                # 计算输出
                optimizer.zero_grad()
                y1, y2, y3, y4 = model(x)  # y1 (batchsize,62)
                # 计算loss
                loss1 = criterion(y1,label1_aa) * lam_aa + criterion(y1,label1_bb) * (1. - lam_aa)
                loss2 = criterion(y2,label2_aa) * lam_aa + criterion(y2,label2_bb) * (1. - lam_aa)
                loss3 = criterion(y3,label3_aa) * lam_aa + criterion(y3,label3_bb) * (1. - lam_aa)
                loss4 = criterion(y4,label4_aa) * lam_aa + criterion(y4,label4_bb) * (1. - lam_aa)
                loss = loss1 + loss2 + loss3 + loss4
            # 使用CutMix数据增强
            elif beta > 0 and r < (cutmix_prob + mixup_prob):
                # beta越大,区间中间出现的概率越高
                lam = np.random.beta(beta,beta)
                # 随机打乱索引顺序
                rand_index = torch.randperm(x.size()[0])
                if torch.cuda.is_available():
                    rand_index =rand_index.cuda()
                # 获取 label
                label_a = label
                label1_a, label2_a, label3_a, label4_a = label_a[:, 0], label_a[:, 1], label_a[:, 2], label_a[:, 3]
                label_b = label[rand_index]
                label1_b, label2_b, label3_b, label4_b = label_b[:, 0], label_b[:, 1], label_b[:, 2], label_b[:, 3]
                bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
                # CutMix制作
                x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
                # 计算mix占比
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
                optimizer.zero_grad()
                y1, y2, y3, y4 = model(x)  # y1 (batchsize,62)
                # 计算Loss
                loss1 = criterion(y1,label1_a) * lam + criterion(y1,label1_b) * (1. - lam)
                loss2 = criterion(y2,label2_a) * lam + criterion(y2,label2_b) * (1. - lam)
                loss3 = criterion(y3,label3_a) * lam + criterion(y3,label3_b) * (1. - lam)
                loss4 = criterion(y4,label4_a) * lam + criterion(y4,label4_b) * (1. - lam)
                loss = loss1 + loss2 + loss3 + loss4
            else:
                # 获取验证码4个标签
                label1, label2 ,label3, label4= label[:, 0], label[:, 1], label[:, 2], label[:, 3]
                optimizer.zero_grad()
                y1, y2, y3, y4 = model(x)
                # 计算损失 CrossEntropyLoss, 记录的是每个 batchsize 上的损失
                loss1, loss2, loss3, loss4 = criterion(y1, label1), criterion(y2, label2), criterion(y3, label3), criterion(y4, label4)
                loss = loss1 + loss2 + loss3 + loss4

            loss_meter.add(loss.item())
            loss.backward()
            optimizer.step()
        if True:
            # one epoch once
            scheduler.step()
            # 验证集准确率
            accuracy,test_loss = test(model, valDataLoader, num_val)
            # if epoch % 5 == 0:
            #     # 计算训练集准确率
            #     accuracy_train, train_loss = test(model, trainDataLoader_val, num_train)
            #     print('train acc:',accuracy_train)
            # 保存最优模型
            if best_acc < accuracy:
                best_acc = accuracy
                # 保存最好的模型
                model.save("_best")
            model.save('_last')
            # -------------------------- 每个epoch输出结果 -----------------------------
            # 输出训练日志
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),end='')
            print("epoch:[%02d/%02d] | Learning rate: %.6f | Train loss %.5f | Test loss %.5f | test acc %.3f | best acc %.3f" % \
                  (epoch,totalEpoch, scheduler.get_last_lr()[0],loss_meter.avg,test_loss, accuracy, best_acc))


# 验证集
def test(model, testDataLoader, num_val):
    model.eval()
    # 验证集图片数目
    totalNum = num_val
    # 预测正确数目
    rightNum = 0
    criterion = nn.CrossEntropyLoss()
    # 记录loss信息
    loss_meter = AverageMeter()
    for circle, (x, label) in enumerate(testDataLoader, 0):
        if torch.cuda.is_available():
            x = x.cuda()
            label = label.cuda()
        label = label.long()
        y1, y2, y3, y4 = model(x)
        # 计算loss
        label1,label2,label3,label4 = label[:, 0], label[:, 1],label[:, 2], label[:, 3]
        loss1, loss2, loss3, loss4 = criterion(y1, label1), criterion(y2, label2), criterion(y3, label3), criterion(y4, label4)
        loss = loss1 + loss2 + loss3 + loss4
        loss_meter.add(loss.item())

        small_bs = x.size()[0]  # 获取batchsize
        y1, y2, y3, y4 = y1.topk(1, dim=1)[1].view(small_bs, 1), \
            y2.topk(1, dim=1)[1].view(small_bs, 1), \
            y3.topk(1, dim=1)[1].view(small_bs, 1), \
            y4.topk(1, dim=1)[1].view(small_bs, 1)
        y = torch.cat((y1, y2, y3, y4), dim=1)
        diff = (y != label)
        diff = diff.sum(1)
        diff = (diff != 0)
        res = diff.sum(0).item()
        rightNum += (small_bs - res)

    return float(rightNum) / float(totalNum) , loss_meter.avg


if __name__ == '__main__':
    net_list = {
                # 'EfficientNet_B0':EfficientNet_B0(),
                # 'EfficientNet_B1':EfficientNet_B1(),
                # 'EfficientNet_B2':EfficientNet_B2(),
                # 'EfficientNet_B3':EfficientNet_B3(),
                # 'EfficientNet_B4':EfficientNet_B4(),
                'EfficientNet_B5':EfficientNet_B5(),
                # 'EfficientNet_B6':EfficientNet_B6(),
                # 'EfficientNet_B7':EfficientNet_B7(),
                # 'EfficientNet_B8':EfficientNet_B8(),
                # 'EfficientNet_B0_CRNN': EfficientNet_B0_CRNN(),
                # 'EfficientNet_B0_sig': EfficientNet_B0_sig(),
                'RepVGG_A0':RepVGG_A0(),
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
    net = net_list['RepVGG_A0']
    # 加载预训练模型
    # net.load_model("./weights/senet_new.pth")
    # 交叉熵损失训练
    train_original(net)
