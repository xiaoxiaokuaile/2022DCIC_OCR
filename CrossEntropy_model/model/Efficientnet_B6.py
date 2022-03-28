# Learner: 王振强
# Learn Time: 2022/2/18 15:17
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
import os


class EfficientNet_B6(nn.Module):
    def __init__(self, class_num=62):
        super(EfficientNet_B6, self).__init__()
        pr_model = EfficientNet.from_pretrained('efficientnet-b6',weights_path='./model/Pretrain_model/efficientnet-b6-c76e70fd.pth')
        self.cnn = nn.Sequential(
            pr_model._conv_stem,   # 56 (2,2) (528,528)->(264,264)
            pr_model._bn0,
            pr_model._blocks[0],   # 32
            pr_model._blocks[1],   # 32
            pr_model._blocks[2],   # 32
            pr_model._blocks[3],   # 40 (2,2) (264,264)->(132,132)
            pr_model._blocks[4],   # 40
            pr_model._blocks[5],   # 40
            pr_model._blocks[6],   # 40
            pr_model._blocks[7],   # 40
            pr_model._blocks[8],   # 40
            pr_model._blocks[9],   # 72 (2,2) (132,132)->(66,66)
            pr_model._blocks[10],  # 72
            pr_model._blocks[11],  # 72
            pr_model._blocks[12],  # 72
            pr_model._blocks[13],  # 72
            pr_model._blocks[14],  # 72
            pr_model._blocks[15],  # 144 (2,2) (66,66)->(33,33)
            pr_model._blocks[16],  # 144
            pr_model._blocks[17],  # 144
            pr_model._blocks[18],  # 144
            pr_model._blocks[19],  # 144
            pr_model._blocks[20],  # 144
            pr_model._blocks[21],  # 144
            pr_model._blocks[22],  # 144
            pr_model._blocks[23],  # 200
            pr_model._blocks[24],  # 200
            pr_model._blocks[25],  # 200
            pr_model._blocks[26],  # 200
            pr_model._blocks[27],  # 200
            pr_model._blocks[28],  # 200
            pr_model._blocks[29],  # 200
            pr_model._blocks[30],  # 200
            pr_model._blocks[31],  # 344 (2,2) (33,33)->(17,17)
            pr_model._blocks[32],  # 344
            pr_model._blocks[33],  # 344
            pr_model._blocks[34],  # 344
            pr_model._blocks[35],  # 344
            pr_model._blocks[36],  # 344
            pr_model._blocks[37],  # 344
            pr_model._blocks[38],  # 344
            pr_model._blocks[39],  # 344
            pr_model._blocks[40],  # 344
            pr_model._blocks[41],  # 344
            pr_model._blocks[42],  # 576
            pr_model._blocks[43],  # 576
            pr_model._blocks[44],  # 576
            pr_model._conv_head,   # (2304,17,17)
            pr_model._bn1,
            pr_model._avg_pooling, # (2304,1,1)
            # pr_model._dropout,     # Dropout(p=0.5, inplace=False)
            # pr_model._fc,
            # pr_model._swish
        )
        # 将特征图转化为 (1,1) 大小
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2304, class_num)
        self.fc2 = nn.Linear(2304, class_num)
        self.fc3 = nn.Linear(2304, class_num)
        self.fc4 = nn.Linear(2304, class_num)

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
        name = "./weights/EfficientNet_B6" + str(circle) + ".pth"
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
    inputs = torch.rand(4, 3, 528, 528)
    model = EfficientNet.from_pretrained('efficientnet-b6',weights_path='./Pretrain_model/efficientnet-b6-c76e70fd.pth')
    print(model)
    model1 = nn.Sequential(
        model._conv_stem,   # 56 (2,2) (528,528)->(264,264)
        model._bn0,
        model._blocks[0],   # 32
        model._blocks[1],   # 32
        model._blocks[2],   # 32
        model._blocks[3],   # 40 (2,2) (264,264)->(132,132)
        model._blocks[4],   # 40
        model._blocks[5],   # 40
        model._blocks[6],   # 40
        model._blocks[7],   # 40
        model._blocks[8],   # 40
        model._blocks[9],   # 72 (2,2) (132,132)->(66,66)
        model._blocks[10],  # 72
        model._blocks[11],  # 72
        model._blocks[12],  # 72
        model._blocks[13],  # 72
        model._blocks[14],  # 72
        model._blocks[15],  # 144 (2,2) (66,66)->(33,33)
        model._blocks[16],  # 144
        model._blocks[17],  # 144
        model._blocks[18],  # 144
        model._blocks[19],  # 144
        model._blocks[20],  # 144
        model._blocks[21],  # 144
        model._blocks[22],  # 144
        model._blocks[23],  # 200
        model._blocks[24],  # 200
        model._blocks[25],  # 200
        model._blocks[26],  # 200
        model._blocks[27],  # 200
        model._blocks[28],  # 200
        model._blocks[29],  # 200
        model._blocks[30],  # 200
        model._blocks[31],  # 344 (2,2) (33,33)->(17,17)
        model._blocks[32],  # 344
        model._blocks[33],  # 344
        model._blocks[34],  # 344
        model._blocks[35],  # 344
        model._blocks[36],  # 344
        model._blocks[37],  # 344
        model._blocks[38],  # 344
        model._blocks[39],  # 344
        model._blocks[40],  # 344
        model._blocks[41],  # 344
        model._blocks[42],  # 576
        model._blocks[43],  # 576
        model._blocks[44],  # 576
        model._conv_head,   # (2304,17,17)
        model._bn1,
        model._avg_pooling, # (2304,1,1)
        # model._dropout,     # Dropout(p=0.5, inplace=False)
        # model._fc,
        # model._swish
    )
    out2 = model1(inputs)
    print(out2.shape)







