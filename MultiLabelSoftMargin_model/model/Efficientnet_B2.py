# Learner: 王振强
# Learn Time: 2022/2/18 15:14
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
import os


class EfficientNet_B2(nn.Module):
    def __init__(self, class_num=248):
        super(EfficientNet_B2, self).__init__()
        pr_model = EfficientNet.from_pretrained('efficientnet-b2',weights_path='./model/Pretrain_model/efficientnet-b2-8bb594d6.pth')
        self.cnn = nn.Sequential(
            pr_model._conv_stem,   # 32 (2,2) (260,260)->(130,130)
            pr_model._bn0,
            pr_model._blocks[0],   # 16
            pr_model._blocks[1],   # 16
            pr_model._blocks[2],   # 24 (2,2) (130,130)->(65,65)
            pr_model._blocks[3],   # 24
            pr_model._blocks[4],   # 24
            pr_model._blocks[5],   # 48 (2,2) (65,65)->(33,33)
            pr_model._blocks[6],   # 48
            pr_model._blocks[7],   # 48
            pr_model._blocks[8],   # 88 (2,2) (33,33)->(17,17)
            pr_model._blocks[9],   # 88
            pr_model._blocks[10],  # 88
            pr_model._blocks[11],  # 88
            pr_model._blocks[12],  # 120 (2,2) (17,17)
            pr_model._blocks[13],  # 120
            pr_model._blocks[14],  # 120
            pr_model._blocks[15],  # 120
            pr_model._blocks[16],  # 208 (2,2) (17,17)->(9,9)
            pr_model._blocks[17],  # 208
            pr_model._blocks[18],  # 208
            pr_model._blocks[19],  # 208
            pr_model._blocks[20],  # 208
            pr_model._blocks[21],  # 352
            pr_model._blocks[22],  # 352
            pr_model._conv_head,   # (1408,9,9)
            pr_model._bn1,
            pr_model._avg_pooling, # (1408,1,1)
            # pr_model._dropout,     # Dropout(p=0.3, inplace=False)
            # pr_model._fc,
            # pr_model._swish
        )
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(1408, class_num)


    def forward(self, x):
        x = self.cnn(x)  # (batchsize,1408)
        # 维度变换
        x = x.view(x.shape[0], -1)
        x = self.drop(x)
        y = self.fc(x)

        return y

    def save(self, circle):
        name = "./weights/EfficientNet_B2" + str(circle) + ".pth"
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
    inputs = torch.rand(32, 3, 260, 260)
    model = EfficientNet.from_pretrained('efficientnet-b2',weights_path='./Pretrain_model/efficientnet-b2-8bb594d6.pth')
    print(model)
    model1 = nn.Sequential(
        model._conv_stem,   # 32 (2,2) (260,260)->(130,130)
        model._bn0,
        model._blocks[0],   # 16
        model._blocks[1],   # 16
        model._blocks[2],   # 24 (2,2) (130,130)->(65,65)
        model._blocks[3],   # 24
        model._blocks[4],   # 24
        model._blocks[5],   # 48 (2,2) (65,65)->(33,33)
        model._blocks[6],   # 48
        model._blocks[7],   # 48
        model._blocks[8],   # 88 (2,2) (33,33)->(17,17)
        model._blocks[9],   # 88
        model._blocks[10],  # 88
        model._blocks[11],  # 88
        model._blocks[12],  # 120 (2,2) (17,17)
        model._blocks[13],  # 120
        model._blocks[14],  # 120
        model._blocks[15],  # 120
        model._blocks[16],  # 208 (2,2) (17,17)->(9,9)
        model._blocks[17],  # 208
        model._blocks[18],  # 208
        model._blocks[19],  # 208
        model._blocks[20],  # 208
        model._blocks[21],  # 352
        model._blocks[22],  # 352
        model._conv_head,   # (1408,9,9)
        model._bn1,
        model._avg_pooling, # (1408,1,1)
        # model._dropout,     # Dropout(p=0.3, inplace=False)
        # model._fc,
        # model._swish
    )
    out2 = model1(inputs)
    print(out2.shape)














