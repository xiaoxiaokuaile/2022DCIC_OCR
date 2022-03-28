# Learner: 王振强
# Learn Time: 2022/2/18 15:18
# Learner: 王振强
# Learn Time: 2022/2/18 15:13
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
import os


class EfficientNet_B8(nn.Module):
    def __init__(self, class_num=62):
        super(EfficientNet_B8, self).__init__()
        pr_model = EfficientNet.from_pretrained('efficientnet-b8',weights_path='./model/Pretrain_model/adv-efficientnet-b8-22a8fe65.pth')
        self.cnn = nn.Sequential(
            pr_model._conv_stem,   # 72 (2,2) (672,672)->(336,336)
            pr_model._bn0,
            pr_model._blocks[0],   # 32
            pr_model._blocks[1],   # 32
            pr_model._blocks[2],   # 32
            pr_model._blocks[3],   # 32
            pr_model._blocks[4],   # 56 (2,2) (336,336)->(168,168)
            pr_model._blocks[5],   # 56
            pr_model._blocks[6],   # 56
            pr_model._blocks[7],   # 56
            pr_model._blocks[8],   # 56
            pr_model._blocks[9],   # 56
            pr_model._blocks[10],  # 56
            pr_model._blocks[11],  # 56
            pr_model._blocks[12],  # 88 (2,2) (168,168)->(84,84)
            pr_model._blocks[13],  # 88
            pr_model._blocks[14],  # 88
            pr_model._blocks[15],  # 88
            pr_model._blocks[16],  # 88
            pr_model._blocks[17],  # 88
            pr_model._blocks[18],  # 88
            pr_model._blocks[19],  # 88
            pr_model._blocks[20],  # 176 (2,2) (84,84)->(42,42)
            pr_model._blocks[21],  # 176
            pr_model._blocks[22],  # 176
            pr_model._blocks[23],  # 176
            pr_model._blocks[24],  # 176
            pr_model._blocks[25],  # 176
            pr_model._blocks[26],  # 176
            pr_model._blocks[27],  # 176
            pr_model._blocks[28],  # 176
            pr_model._blocks[29],  # 176
            pr_model._blocks[30],  # 176
            pr_model._blocks[31],  # 248
            pr_model._blocks[32],  # 248
            pr_model._blocks[33],  # 248
            pr_model._blocks[34],  # 248
            pr_model._blocks[35],  # 248
            pr_model._blocks[36],  # 248
            pr_model._blocks[37],  # 248
            pr_model._blocks[38],  # 248
            pr_model._blocks[39],  # 248
            pr_model._blocks[40],  # 248
            pr_model._blocks[41],  # 248
            pr_model._blocks[42],  # 424 (2,2) (42,42)->(21,21)
            pr_model._blocks[43],  # 424
            pr_model._blocks[44],  # 424
            pr_model._blocks[45],  # 424
            pr_model._blocks[46],  # 424
            pr_model._blocks[47],  # 424
            pr_model._blocks[48],  # 424
            pr_model._blocks[49],  # 424
            pr_model._blocks[50],  # 424
            pr_model._blocks[51],  # 424
            pr_model._blocks[52],  # 424
            pr_model._blocks[53],  # 424
            pr_model._blocks[54],  # 424
            pr_model._blocks[55],  # 424
            pr_model._blocks[56],  # 424
            pr_model._blocks[57],  # 704
            pr_model._blocks[58],  # 704
            pr_model._blocks[59],  # 704
            pr_model._blocks[60],  # 704
            pr_model._conv_head,   # (2816,21,21)
            pr_model._bn1,
            pr_model._avg_pooling, # (2816,1,1)
            # pr_model._dropout,     # Dropout(p=0.5, inplace=False)
            # pr_model._fc,
            # pr_model._swish
        )
        # 将特征图转化为(1,1)大小
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2816, class_num)
        self.fc2 = nn.Linear(2816, class_num)
        self.fc3 = nn.Linear(2816, class_num)
        self.fc4 = nn.Linear(2816, class_num)

    def forward(self, x):
        x = self.cnn(x)  # (batchsize,1280)
        # 维度变换
        x = x.view(x.shape[0], -1)
        x = self.drop(x)
        y1 = self.fc1(x)
        y2 = self.fc2(x)
        y3 = self.fc3(x)
        y4 = self.fc4(x)
        return y1, y2, y3, y4

    def save(self, circle):
        name = "./weights/EfficientNet_B8" + str(circle) + ".pth"
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
    inputs = torch.rand(1, 3, 672, 672)
    model = EfficientNet.from_pretrained('efficientnet-b8',weights_path='./Pretrain_model/adv-efficientnet-b8-22a8fe65.pth')
    print(model)
    model1 = nn.Sequential(
        model._conv_stem,   # 72 (2,2) (672,672)->(336,336)
        model._bn0,
        model._blocks[0],   # 32
        model._blocks[1],   # 32
        model._blocks[2],   # 32
        model._blocks[3],   # 32
        model._blocks[4],   # 56 (2,2) (336,336)->(168,168)
        model._blocks[5],   # 56
        model._blocks[6],   # 56
        model._blocks[7],   # 56
        model._blocks[8],   # 56
        model._blocks[9],   # 56
        model._blocks[10],  # 56
        model._blocks[11],  # 56
        model._blocks[12],  # 88 (2,2) (168,168)->(84,84)
        model._blocks[13],  # 88
        model._blocks[14],  # 88
        model._blocks[15],  # 88
        model._blocks[16],  # 88
        model._blocks[17],  # 88
        model._blocks[18],  # 88
        model._blocks[19],  # 88
        model._blocks[20],  # 176 (2,2) (84,84)->(42,42)
        model._blocks[21],  # 176
        model._blocks[22],  # 176
        model._blocks[23],  # 176
        model._blocks[24],  # 176
        model._blocks[25],  # 176
        model._blocks[26],  # 176
        model._blocks[27],  # 176
        model._blocks[28],  # 176
        model._blocks[29],  # 176
        model._blocks[30],  # 176
        model._blocks[31],  # 248
        model._blocks[32],  # 248
        model._blocks[33],  # 248
        model._blocks[34],  # 248
        model._blocks[35],  # 248
        model._blocks[36],  # 248
        model._blocks[37],  # 248
        model._blocks[38],  # 248
        model._blocks[39],  # 248
        model._blocks[40],  # 248
        model._blocks[41],  # 248
        model._blocks[42],  # 424 (2,2) (42,42)->(21,21)
        model._blocks[43],  # 424
        model._blocks[44],  # 424
        model._blocks[45],  # 424
        model._blocks[46],  # 424
        model._blocks[47],  # 424
        model._blocks[48],  # 424
        model._blocks[49],  # 424
        model._blocks[50],  # 424
        model._blocks[51],  # 424
        model._blocks[52],  # 424
        model._blocks[53],  # 424
        model._blocks[54],  # 424
        model._blocks[55],  # 424
        model._blocks[56],  # 424
        model._blocks[57],  # 704
        model._blocks[58],  # 704
        model._blocks[59],  # 704
        model._blocks[60],  # 704
        model._conv_head,   # (2816,21,21)
        model._bn1,
        model._avg_pooling, # (2816,1,1)
        # model._dropout,     # Dropout(p=0.5, inplace=False)
        # model._fc,
        # model._swish
    )
    out2 = model1(inputs)
    print(out2.shape)














