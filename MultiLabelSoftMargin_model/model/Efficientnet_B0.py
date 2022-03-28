# Learner: 王振强
# Learn Time: 2022/2/16 16:35
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
import os


class EfficientNet_B0(nn.Module):
    def __init__(self, class_num=248):
        super(EfficientNet_B0, self).__init__()
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
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(1280, class_num)


    def forward(self, x):
        x = self.cnn(x)  # (batchsize,1280)
        # 维度变换
        x = x.view(x.shape[0], -1)
        x = self.drop(x)
        y = self.fc(x)

        return y

    def save(self, circle):
        name = "./weights/EfficientNet_B0" + str(circle) + ".pth"
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
    # ----------------- 维度变换 -------------------
    # torch.squeeze() 维度压缩,将其中size为1的维度都删除
    # out2 = torch.squeeze(out2)
    # 维度变换
    out2 = out2.view(out2.shape[0],-1)
    print(out2.shape)
    # -------------------------------------------
    out2 = nn.Dropout(0.5)(out2)
    out2 = nn.Linear(in_features=out2.shape[1], out_features=62)(out2)
    print(out2.shape)

    # layer = model.
    feature = model._fc.in_features  # 通道数1280
    print(feature)
    model._fc = nn.Linear(in_features=feature,out_features=45)
    #print(model)
    # net = timm.create_model('tf_efficientnet_b0_ns',pretrained=True)
    outputs = model(inputs)
    # for i in range(len(outputs)):
    #     print('feature {} shape: {}'.format(i, outputs[i].shape))
    model.eval()

    print(outputs.shape)














