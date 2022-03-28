# Learner: 王振强
# Learn Time: 2022/2/27 16:45
# Learner: 王振强
# Learn Time: 2022/2/16 16:35
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
import os


class EfficientNet_B0_sig(nn.Module):
    def __init__(self, class_num=62):
        super(EfficientNet_B0_sig, self).__init__()
        pr_model = EfficientNet.from_pretrained('efficientnet-b0',weights_path='./model/Pretrain_model/efficientnet-b0-355c32eb.pth')
        self.cnn = nn.Sequential(
            pr_model._conv_stem,   # 32 (2,2) (128,320)->(64,160)
            pr_model._bn0,
            pr_model._blocks[0],   # 16
            pr_model._blocks[1],   # 24 (2,2) (64,160)->(32,80)
            pr_model._blocks[2],   # 24
            pr_model._blocks[3],   # 40 (2,2) (32,80)->(16,40)
            pr_model._blocks[4],   # 40
            pr_model._blocks[5],   # 80 (2,2) (16,40)->(8,20)
            pr_model._blocks[6],   # 80
            pr_model._blocks[7],   # 80
            pr_model._blocks[8],   # 112
            pr_model._blocks[9],   # 112
            pr_model._blocks[10],  # 112
            pr_model._blocks[11],  # 192
            pr_model._blocks[12],  # 192 (2,2) (8,20)->(4,10)
            pr_model._blocks[13],  # 192
            pr_model._blocks[14],  # 192
            pr_model._blocks[15],  # 320 (4,10)

            pr_model._conv_head,   # (1280,7,7)
            pr_model._bn1,
            pr_model._avg_pooling, # (1280,1,1)

            # pr_model._dropout,     # Dropout(p=0.2, inplace=False)
            # pr_model._fc,
            # pr_model._swish
        )
        # 将特征图转化为(1,1)大小
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_1 = nn.Sequential(
            nn.Linear(1280, 1280),
            nn.Sigmoid()
        )
        self.drop = nn.Dropout(0.2)
        self.fc1 = nn.Linear(1280, class_num)
        self.fc2 = nn.Linear(1280, class_num)
        self.fc3 = nn.Linear(1280, class_num)
        self.fc4 = nn.Linear(1280, class_num)

    def forward(self, x):
        x = self.cnn(x)  # (batchsize,1280)
        # 维度变换
        x = x.view(x.shape[0], -1)
        # 第一个FC层
        x = self.fc_1(x)
        x = self.drop(x)
        y1 = self.fc1(x)
        y2 = self.fc2(x)
        y3 = self.fc3(x)
        y4 = self.fc4(x)
        return y1, y2, y3, y4

    def save(self, circle):
        name = "./weights/EfficientNet_B0_sig" + str(circle) + ".pth"
        torch.save(self.state_dict(), name)

    # 加载模型
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
    inputs = torch.rand(32, 3, 240, 240)
    model = EfficientNet.from_pretrained('efficientnet-b1',weights_path='./Pretrain_model/efficientnet-b1-f1951068.pth')
    print(model)
    model1 = nn.Sequential(
        model._conv_stem,   # 32 (2,2) (224,224)->(112,112)
        model._bn0,
        model._blocks[0],   # 16
        model._blocks[1],   # 24 (2,2) (112,112)->(56,56)
        model._blocks[2],   # 24
        model._blocks[3],   # 40 (2,2) (56,56)->(28,28)
        model._blocks[4],   # 40
        model._blocks[5],   # 80 (2,2) (28,28)->(14,14)
        model._blocks[6],   # 80
        model._blocks[7],   # 80
        model._blocks[8],   # 112
        model._blocks[9],   # 112
        model._blocks[10],  # 112
        model._blocks[11],  # 192
        model._blocks[12],  # 192 (2,2) (14,14)->(7,7)
        model._blocks[13],  # 192
        model._blocks[14],  # 192
        model._blocks[15],  # 320
        # model._conv_head,   # (1280,7,7)
        # model._bn1,
        # model._avg_pooling, # (1280,1,1)
        # model._dropout,     # Dropout(p=0.2, inplace=False)
        # model._fc,
        # model._swish
    )
    out2 = model1(inputs)
    print(out2.shape)
















