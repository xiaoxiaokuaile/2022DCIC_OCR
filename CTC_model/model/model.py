# Learner: 王振强
# Learn Time: 2022/2/3 23:29
import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

class Swish(nn.Module):
    def forward(self,x):
        return x*torch.sigmoid(x)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Model_resnet18(nn.Module):
    def __init__(self, model_arch, n_classes):
        super(Model_resnet18, self).__init__()
        # pretrained=True 那么timm会从对应的URL下载模型权重参数并载入模型
        model_ft = timm.create_model(model_arch, pretrained=True, num_classes=n_classes)

        self.cnn = nn.Sequential(
            model_ft.conv1,    # [2, 3, 64, 160]
            model_ft.bn1,
            model_ft.act1,
            model_ft.maxpool,  # [2, 64, 32, 80]
            model_ft.layer1,   # [2, 64, 16, 40]
            # model_ft.layer2,   # [2, 128, 8, 20]
            # model_ft.layer3,   # [2, 256, 4, 10]
            # model_ft.layer4,   # [2, 512, 2, 5]
        )

        self.stage = nn.Sequential(
            # # ================== 第2层 ================== (2,2) [2, 128, 16, 40]
            # nn.Conv2d(64, 128, (3, 3), stride=(1, 1), padding=1),
            # nn.BatchNorm2d(128),
            # # -------------- 激活函数 ---------------
            # # nn.LeakyReLU(negative_slope=0.1),
            # Swish(),
            # # ---------------------------------------
            # nn.MaxPool2d((2, 2), stride=(2, 2), padding=0),

            # ================== 第3层 ================== (2,2) [2, 256, 8, 20]
            # nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=1),
            # nn.BatchNorm2d(128),
            # # -------------- 激活函数 ---------------
            # # nn.LeakyReLU(negative_slope=0.1),
            # Swish(),
            # # ---------------------------------------
            # nn.MaxPool2d((2, 2), stride=(2, 2), padding=0),

            # ================== 第4层 ================== (2,2) [2, 256, 4, 10]
            nn.Conv2d(64, 128, (5, 5), stride=(1, 1), padding=2),
            nn.BatchNorm2d(128),
            # -------------- 激活函数 ---------------
            # nn.LeakyReLU(negative_slope=0.1),
            Swish(),
            # ---------------------------------------
            nn.MaxPool2d((2, 2), stride=(2, 2), padding=0),
        )

        self.fc = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=n_classes),
            # nn.Softmax()
        )

    def forward(self, x):
        # (batchsize,3,128,320)->(batchsize,512,4,10)
        # resnet18前3个块
        x = self.cnn(x)
        # (3,3) 卷积
        x = self.stage(x)
        # (batchsize,512,4,10)->(batchsize,2048,10)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        # 维度换位 (10,batchsize,2048)
        x = x.permute(2, 0, 1)
        # (8,batchsize,2048) 因为双向传播所以输出1024*2
        x = self.fc(x)
        return x


class Model_resnet101(nn.Module):
    def __init__(self, model_arch, n_classes):
        super(Model_resnet101, self).__init__()

        #
        # pretrained=True 那么timm会从对应的URL下载模型权重参数并载入模型
        model_ft = timm.create_model(model_arch, pretrained=True, num_classes=n_classes)

        self.cnn = nn.Sequential(
            model_ft.conv1,    # [2, 3, 128, 320]
            model_ft.bn1,
            model_ft.act1,
            model_ft.maxpool,  # [2, 64, 64, 160]
            model_ft.layer1,   # [2, 64, 32, 80]
            model_ft.layer2,   # [2, 128, 16, 40]
            model_ft.layer3,   # [2, 256, 8, 20]
            model_ft.layer4,   # [2, 512, 4, 10]
        )

        # bidirectional是否双向传播
        self.lstm = nn.LSTM(input_size=512 * 4, hidden_size=512, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(in_features=1024, out_features=n_classes)

    def forward(self, x):
        # (batchsize,3,128,256)->(batchsize,2048,4,8)
        x = self.cnn(x)
        # (batchsize,2048,4,8)->(batchsize,8192,8)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        # 维度换位 (8,batchsize,8192)
        x = x.permute(2, 0, 1)
        # (8,batchsize,2048) 因为双向传播所以输出1024*2
        x, _ = self.lstm(x)
        x = F.dropout(x, p=0.5)
        x = self.fc(x)
        return x


