# Learner: 王振强
# Learn Time: 2022/2/18 15:14
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
import os


class EfficientNet_B3(nn.Module):
    def __init__(self, class_num=248):
        super(EfficientNet_B3, self).__init__()
        pr_model = EfficientNet.from_pretrained('efficientnet-b3',weights_path='./model/Pretrain_model/efficientnet-b3-5fb5a3c3.pth')
        self.cnn = nn.Sequential(
            pr_model._conv_stem,   # 32 (2,2) (300,300)->(150,150)
            pr_model._bn0,
            pr_model._blocks[0],   # 24
            pr_model._blocks[1],   # 24
            pr_model._blocks[2],   # 32 (2,2) (150,150)->(75,75)
            pr_model._blocks[3],   # 32
            pr_model._blocks[4],   # 32
            pr_model._blocks[5],   # 48 (2,2) (75,75)->(38,38)
            pr_model._blocks[6],   # 48
            pr_model._blocks[7],   # 48
            pr_model._blocks[8],   # 48
            pr_model._blocks[9],   # 96 (2,2) (38,38)->(19,19)
            pr_model._blocks[10],  # 96
            pr_model._blocks[11],  # 96
            pr_model._blocks[12],  # 96
            pr_model._blocks[13],  # 136
            pr_model._blocks[14],  # 136
            pr_model._blocks[15],  # 136
            pr_model._blocks[16],  # 136
            pr_model._blocks[17],  # 136
            pr_model._blocks[18],  # 232 (2,2) (10,10)
            pr_model._blocks[19],  # 232
            pr_model._blocks[20],  # 232
            pr_model._blocks[21],  # 232
            pr_model._blocks[22],  # 232
            pr_model._blocks[23],  # 232
            pr_model._blocks[24],  # 384
            pr_model._blocks[25],  # 384
            pr_model._conv_head,   # (1536,7,7)
            pr_model._bn1,
            pr_model._avg_pooling, # (1536,1,1)
            # pr_model._dropout,     # Dropout(p=0.3, inplace=False)
            # pr_model._fc,
            # pr_model._swish
        )
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(1536, class_num)

    def forward(self, x):
        x = self.cnn(x)  # (batchsize,1280)
        # 维度变换
        x = x.view(x.shape[0], -1)
        x = self.drop(x)
        y = self.fc(x)

        return y

    def save(self, circle):
        name = "./weights/EfficientNet_B3" + str(circle) + ".pth"
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
    inputs = torch.rand(8, 3, 300, 300)
    model = EfficientNet.from_pretrained('efficientnet-b3',weights_path='./Pretrain_model/efficientnet-b3-5fb5a3c3.pth')
    print(model)
    model1 = nn.Sequential(
        model._conv_stem,   # 32 (2,2) (300,300)->(150,150)
        model._bn0,
        model._blocks[0],   # 24
        model._blocks[1],   # 24
        model._blocks[2],   # 32 (2,2) (150,150)->(75,75)
        model._blocks[3],   # 32
        model._blocks[4],   # 32
        model._blocks[5],   # 48 (2,2) (75,75)->(38,38)
        model._blocks[6],   # 48
        model._blocks[7],   # 48
        model._blocks[8],   # 48
        model._blocks[9],   # 96 (2,2) (38,38)->(19,19)
        model._blocks[10],  # 96
        model._blocks[11],  # 96
        model._blocks[12],  # 96
        model._blocks[13],  # 136
        model._blocks[14],  # 136
        model._blocks[15],  # 136
        model._blocks[16],  # 136
        model._blocks[17],  # 136
        model._blocks[18],  # 232 (2,2) (10,10)
        model._blocks[19],  # 232
        model._blocks[20],  # 232
        model._blocks[21],  # 232
        model._blocks[22],  # 232
        model._blocks[23],  # 232
        model._blocks[24],  # 384
        model._blocks[25],  # 384
        model._conv_head,   # (1536,7,7)
        model._bn1,
        model._avg_pooling, # (1536,1,1)
        # model._dropout,     # Dropout(p=0.2, inplace=False)
        # model._fc,
        # model._swish
    )
    out2 = model1(inputs)
    print(out2.shape)















