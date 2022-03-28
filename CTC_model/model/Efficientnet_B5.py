# Learner: 王振强
# Learn Time: 2022/2/18 15:17
# Learner: 王振强
# Learn Time: 2022/2/18 15:13
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
import os


class EfficientNet_B5(nn.Module):
    def __init__(self, class_num=63):
        super(EfficientNet_B5, self).__init__()
        pr_model = EfficientNet.from_pretrained('efficientnet-b5',weights_path='./model/Pretrain_model/efficientnet-b5-b6417697.pth')
        self.cnn = nn.Sequential(
            pr_model._conv_stem,   # 48 (2,2) (128,320)->(64,160)
            pr_model._bn0,
            pr_model._blocks[0],   # 24
            pr_model._blocks[1],   # 24
            pr_model._blocks[2],   # 24
            pr_model._blocks[3],   # 40 (2,2) (64,160)->(32,80)
            pr_model._blocks[4],   # 40
            pr_model._blocks[5],   # 40
            pr_model._blocks[6],   # 40
            pr_model._blocks[7],   # 40
            pr_model._blocks[8],   # 64 (2,2) (32,80)->(16,40)
            pr_model._blocks[9],   # 64
            pr_model._blocks[10],  # 64
            pr_model._blocks[11],  # 64
            pr_model._blocks[12],  # 64
            pr_model._blocks[13],  # 128 (2,2) (16,40)->(8,20)
            pr_model._blocks[14],  # 128
            pr_model._blocks[15],  # 128
            pr_model._blocks[16],  # 128
            pr_model._blocks[17],  # 128
            pr_model._blocks[18],  # 128
            pr_model._blocks[19],  # 128
            pr_model._blocks[20],  # 176
            pr_model._blocks[21],  # 176
            pr_model._blocks[22],  # 176
            pr_model._blocks[23],  # 176
            pr_model._blocks[24],  # 176
            pr_model._blocks[25],  # 176
            pr_model._blocks[26],  # 176
            pr_model._blocks[27],  # 304 (2,2) (8,20)->(4,10)
            pr_model._blocks[28],  # 304
            pr_model._blocks[29],  # 304
            pr_model._blocks[30],  # 304
            pr_model._blocks[31],  # 304
            pr_model._blocks[32],  # 304
            pr_model._blocks[33],  # 304
            pr_model._blocks[34],  # 304
            pr_model._blocks[35],  # 304
            pr_model._blocks[36],  # 512
            pr_model._blocks[37],  # 512
            pr_model._blocks[38],  # 512
            # pr_model._conv_head,   # (2048,15,15)
            # pr_model._bn1,
            # pr_model._avg_pooling, # (2048,1,1)
            # pr_model._dropout,     # Dropout(p=0.2, inplace=False)
            # pr_model._fc,
            # pr_model._swish
        )
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(in_features=2048, out_features=class_num),
            # nn.Sigmoid()
            # nn.Softmax()
        )

    def forward(self, x):
        # (batchsize,3,128,320)->(batchsize,512,4,10)
        x = self.cnn(x)
        # (batchsize,512,4,10)->(batchsize,2048,10)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        # 维度换位 (10,batchsize,2048)
        x = x.permute(2, 0, 1)
        # (10,batchsize,2048)
        x = self.fc(x)

        return x

    def save(self, circle):
        name = "./weights/EfficientNet_B5" + str(circle) + ".pth"
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
    inputs = torch.rand(4, 3, 456, 456)
    model = EfficientNet.from_pretrained('efficientnet-b5',weights_path='./Pretrain_model/efficientnet-b5-b6417697.pth')
    print(model)
    model1 = nn.Sequential(
        model._conv_stem,   # 48 (2,2) (456,456)->(228,228)
        model._bn0,
        model._blocks[0],   # 24
        model._blocks[1],   # 24
        model._blocks[2],   # 24
        model._blocks[3],   # 40 (2,2) (228,228)->(114,114)
        model._blocks[4],   # 40
        model._blocks[5],   # 40
        model._blocks[6],   # 40
        model._blocks[7],   # 40
        model._blocks[8],   # 64 (2,2) (114,114)->(57,57)
        model._blocks[9],   # 64
        model._blocks[10],  # 64
        model._blocks[11],  # 64
        model._blocks[12],  # 64
        model._blocks[13],  # 128 (2,2) (57,57)->(29,29)
        model._blocks[14],  # 128
        model._blocks[15],  # 128
        model._blocks[16],  # 128
        model._blocks[17],  # 128
        model._blocks[18],  # 128
        model._blocks[19],  # 128
        model._blocks[20],  # 176
        model._blocks[21],  # 176
        model._blocks[22],  # 176
        model._blocks[23],  # 176
        model._blocks[24],  # 176
        model._blocks[25],  # 176
        model._blocks[26],  # 176
        model._blocks[27],  # 304 (2,2) (29,29)->(15,15)
        model._blocks[28],  # 304
        model._blocks[29],  # 304
        model._blocks[30],  # 304
        model._blocks[31],  # 304
        model._blocks[32],  # 304
        model._blocks[33],  # 304
        model._blocks[34],  # 304
        model._blocks[35],  # 304
        model._blocks[36],  # 512
        model._blocks[37],  # 512
        model._blocks[38],  # 512
        model._conv_head,   # (2048,15,15)
        model._bn1,
        model._avg_pooling, # (2048,1,1)
        # model._dropout,     # Dropout(p=0.2, inplace=False)
        # model._fc,
        # model._swish
    )
    out2 = model1(inputs)
    print(out2.shape)














