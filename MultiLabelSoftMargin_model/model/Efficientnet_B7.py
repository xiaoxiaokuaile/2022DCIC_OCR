# Learner: 王振强
# Learn Time: 2022/2/18 15:18
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
import os


class EfficientNet_B7(nn.Module):
    def __init__(self, class_num=248):
        super(EfficientNet_B7, self).__init__()
        pr_model = EfficientNet.from_pretrained('efficientnet-b7',weights_path='./model/Pretrain_model/efficientnet-b7-dcc49843.pth')
        self.cnn = nn.Sequential(
            pr_model._conv_stem,   # 64 (2,2) (600,600)->(300,300)
            pr_model._bn0,
            pr_model._blocks[0],   # 32
            pr_model._blocks[1],   # 32
            pr_model._blocks[2],   # 32
            pr_model._blocks[3],   # 32
            pr_model._blocks[4],   # 48 (2,2) (300,300)->(150,150)
            pr_model._blocks[5],   # 48
            pr_model._blocks[6],   # 48
            pr_model._blocks[7],   # 48
            pr_model._blocks[8],   # 48
            pr_model._blocks[9],   # 48
            pr_model._blocks[10],  # 48
            pr_model._blocks[11],  # 80 (2,2) (150,150)->(75,75)
            pr_model._blocks[12],  # 80
            pr_model._blocks[13],  # 80
            pr_model._blocks[14],  # 80
            pr_model._blocks[15],  # 80
            pr_model._blocks[16],  # 80
            pr_model._blocks[17],  # 80
            pr_model._blocks[18],  # 160 (2,2) (75,75)->(38,38)
            pr_model._blocks[19],  # 160
            pr_model._blocks[20],  # 160
            pr_model._blocks[21],  # 160
            pr_model._blocks[22],  # 160
            pr_model._blocks[23],  # 160
            pr_model._blocks[24],  # 160
            pr_model._blocks[25],  # 160
            pr_model._blocks[26],  # 160
            pr_model._blocks[27],  # 160
            pr_model._blocks[28],  # 224
            pr_model._blocks[29],  # 224
            pr_model._blocks[30],  # 224
            pr_model._blocks[31],  # 224
            pr_model._blocks[32],  # 224
            pr_model._blocks[33],  # 224
            pr_model._blocks[34],  # 224
            pr_model._blocks[35],  # 224
            pr_model._blocks[36],  # 224
            pr_model._blocks[37],  # 224
            pr_model._blocks[38],  # 384 (2,2) (38,38)->(19,19)
            pr_model._blocks[39],  # 384
            pr_model._blocks[40],  # 384
            pr_model._blocks[41],  # 384
            pr_model._blocks[42],  # 384
            pr_model._blocks[43],  # 384
            pr_model._blocks[44],  # 384
            pr_model._blocks[45],  # 384
            pr_model._blocks[46],  # 384
            pr_model._blocks[47],  # 384
            pr_model._blocks[48],  # 384
            pr_model._blocks[49],  # 384
            pr_model._blocks[50],  # 384
            pr_model._blocks[51],  # 640
            pr_model._blocks[52],  # 640
            pr_model._blocks[53],  # 640
            pr_model._blocks[54],  # 640
            pr_model._conv_head,   # (2560,19,19)
            pr_model._bn1,
            pr_model._avg_pooling, # (2560,1,1)
            # pr_model._dropout,     # Dropout(p=0.5, inplace=False)
            # pr_model._fc,
            # pr_model._swish
        )
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(2560, class_num)

    def forward(self, x):
        x = self.cnn(x)  # (batchsize,1280)
        # 维度变换
        x = x.view(x.shape[0], -1)
        x = self.drop(x)
        y = self.fc(x)
        return y

    def save(self, circle):
        name = "./weights/EfficientNet_B7" + str(circle) + ".pth"
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
    inputs = torch.rand(4, 3, 600, 600)
    model = EfficientNet.from_pretrained('efficientnet-b7',weights_path='./Pretrain_model/efficientnet-b7-dcc49843.pth')
    print(model)
    model1 = nn.Sequential(
        model._conv_stem,   # 64 (2,2) (600,600)->(300,300)
        model._bn0,
        model._blocks[0],   # 32
        model._blocks[1],   # 32
        model._blocks[2],   # 32
        model._blocks[3],   # 32
        model._blocks[4],   # 48 (2,2) (300,300)->(150,150)
        model._blocks[5],   # 48
        model._blocks[6],   # 48
        model._blocks[7],   # 48
        model._blocks[8],   # 48
        model._blocks[9],   # 48
        model._blocks[10],  # 48
        model._blocks[11],  # 80 (2,2) (150,150)->(75,75)
        model._blocks[12],  # 80
        model._blocks[13],  # 80
        model._blocks[14],  # 80
        model._blocks[15],  # 80
        model._blocks[16],  # 80
        model._blocks[17],  # 80
        model._blocks[18],  # 160 (2,2) (75,75)->(38,38)
        model._blocks[19],  # 160
        model._blocks[20],  # 160
        model._blocks[21],  # 160
        model._blocks[22],  # 160
        model._blocks[23],  # 160
        model._blocks[24],  # 160
        model._blocks[25],  # 160
        model._blocks[26],  # 160
        model._blocks[27],  # 160
        model._blocks[28],  # 224
        model._blocks[29],  # 224
        model._blocks[30],  # 224
        model._blocks[31],  # 224
        model._blocks[32],  # 224
        model._blocks[33],  # 224
        model._blocks[34],  # 224
        model._blocks[35],  # 224
        model._blocks[36],  # 224
        model._blocks[37],  # 224
        model._blocks[38],  # 384 (2,2) (38,38)->(19,19)
        model._blocks[39],  # 384
        model._blocks[40],  # 384
        model._blocks[41],  # 384
        model._blocks[42],  # 384
        model._blocks[43],  # 384
        model._blocks[44],  # 384
        model._blocks[45],  # 384
        model._blocks[46],  # 384
        model._blocks[47],  # 384
        model._blocks[48],  # 384
        model._blocks[49],  # 384
        model._blocks[50],  # 384
        model._blocks[51],  # 640
        model._blocks[52],  # 640
        model._blocks[53],  # 640
        model._blocks[54],  # 640
        model._conv_head,   # (2560,19,19)
        model._bn1,
        model._avg_pooling, # (2560,1,1)
        # model._dropout,     # Dropout(p=0.2, inplace=False)
        # model._fc,
        # model._swish
    )
    out2 = model1(inputs)
    print(out2.shape)







