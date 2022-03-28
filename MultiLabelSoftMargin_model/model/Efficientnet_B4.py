# Learner: 王振强
# Learn Time: 2022/2/18 15:14
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
import os


class EfficientNet_B4(nn.Module):
    def __init__(self, class_num=248):
        super(EfficientNet_B4, self).__init__()
        pr_model = EfficientNet.from_pretrained('efficientnet-b4',weights_path='./model/Pretrain_model/efficientnet-b4-6ed6700e.pth')
        self.cnn = nn.Sequential(
            pr_model._conv_stem,   # 48 (2,2) (380,380)->(190,190)
            pr_model._bn0,
            pr_model._blocks[0],   # 24
            pr_model._blocks[1],   # 24
            pr_model._blocks[2],   # 32 (2,2) (190,190)->(95,95)
            pr_model._blocks[3],   # 32
            pr_model._blocks[4],   # 32
            pr_model._blocks[5],   # 32
            pr_model._blocks[6],   # 56 (2,2) (95,95)->(48,48)
            pr_model._blocks[7],   # 56
            pr_model._blocks[8],   # 56
            pr_model._blocks[9],   # 56
            pr_model._blocks[10],  # 112 (2,2) (48,48)->(24,24)
            pr_model._blocks[11],  # 112
            pr_model._blocks[12],  # 112
            pr_model._blocks[13],  # 112
            pr_model._blocks[14],  # 112
            pr_model._blocks[15],  # 112
            pr_model._blocks[16],  # 160
            pr_model._blocks[17],  # 160
            pr_model._blocks[18],  # 160
            pr_model._blocks[19],  # 160
            pr_model._blocks[20],  # 160
            pr_model._blocks[21],  # 160
            pr_model._blocks[22],  # 272 (2,2) (24,24)->(12,12)
            pr_model._blocks[23],  # 272
            pr_model._blocks[24],  # 272
            pr_model._blocks[25],  # 272
            pr_model._blocks[26],  # 272
            pr_model._blocks[27],  # 272
            pr_model._blocks[28],  # 272
            pr_model._blocks[29],  # 272
            pr_model._blocks[30],  # 448
            pr_model._blocks[31],  # 448
            pr_model._conv_head,   # (1792,12,12)
            pr_model._bn1,
            pr_model._avg_pooling, # (1792,1,1)
            # pr_model._dropout,     # Dropout(p=0.4, inplace=False)
            # pr_model._fc,
            # pr_model._swish
        )
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Linear(1792, class_num)

    def forward(self, x):
        x = self.cnn(x)  # (batchsize,1280)
        # 维度变换
        x = x.view(x.shape[0], -1)
        x = self.drop(x)
        y = self.fc(x)

        return y

    def save(self, circle):
        name = "./weights/EfficientNet_B4" + str(circle) + ".pth"
        torch.save(self.state_dict(), name)

    def load_model(self, weight_path):
        if os.path.isfile(weight_path):
            if torch.cuda.is_available():
                self.load_state_dict(torch.load(weight_path))
            else:
                self.load_state_dict(torch.load(weight_path, map_location='cpu'))
            print("load %s success!" % weight_path)
        else:
            print("%s do not exists." % weight_path)


if __name__ == '__main__':
    inputs = torch.rand(4, 3, 380, 380)
    model = EfficientNet.from_pretrained('efficientnet-b4',weights_path='./Pretrain_model/efficientnet-b4-6ed6700e.pth')
    print(model)
    model1 = nn.Sequential(
        model._conv_stem,   # 48 (2,2) (380,380)->(190,190)
        model._bn0,
        model._blocks[0],   # 24
        model._blocks[1],   # 24
        model._blocks[2],   # 32 (2,2) (190,190)->(95,95)
        model._blocks[3],   # 32
        model._blocks[4],   # 32
        model._blocks[5],   # 32
        model._blocks[6],   # 56 (2,2) (95,95)->(48,48)
        model._blocks[7],   # 56
        model._blocks[8],   # 56
        model._blocks[9],   # 56
        model._blocks[10],  # 112 (2,2) (48,48)->(24,24)
        model._blocks[11],  # 112
        model._blocks[12],  # 112
        model._blocks[13],  # 112
        model._blocks[14],  # 112
        model._blocks[15],  # 112
        model._blocks[16],  # 160
        model._blocks[17],  # 160
        model._blocks[18],  # 160
        model._blocks[19],  # 160
        model._blocks[20],  # 160
        model._blocks[21],  # 160
        model._blocks[22],  # 272 (2,2) (24,24)->(12,12)
        model._blocks[23],  # 272
        model._blocks[24],  # 272
        model._blocks[25],  # 272
        model._blocks[26],  # 272
        model._blocks[27],  # 272
        model._blocks[28],  # 272
        model._blocks[29],  # 272
        model._blocks[30],  # 448
        model._blocks[31],  # 448
        model._conv_head,   # (1792,12,12)
        model._bn1,
        model._avg_pooling, # (1792,1,1)
        # model._dropout,     # Dropout(p=0.4, inplace=False)
        # model._fc,
        # model._swish
    )
    out2 = model1(inputs)
    print(out2.shape)
















