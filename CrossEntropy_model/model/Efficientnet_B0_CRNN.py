# Learner: 王振强
# Learn Time: 2022/2/16 16:35
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
import os


class EfficientNet_B0_CRNN(nn.Module):
    def __init__(self, class_num=62):
        super(EfficientNet_B0_CRNN, self).__init__()
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
            pr_model._blocks[15],  # 320 (4,10) [bz,320,4,10]
        )
        # bidirectional是否双向传播
        self.lstm = nn.LSTM(input_size=320 * 4, hidden_size=256, num_layers=2, bidirectional=True)

        self.drop = nn.Dropout(0.2)
        self.fc1 = nn.Linear(5120, class_num)
        self.fc2 = nn.Linear(5120, class_num)
        self.fc3 = nn.Linear(5120, class_num)
        self.fc4 = nn.Linear(5120, class_num)

    def forward(self, x):
        x = self.cnn(x)  # (batchsize,1280)
        # 维度变换
        # (batchsize,320,4,10)->(batchsize,1280,10)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        # 维度换位 (10,batchsize,1280)
        x = x.permute(2, 0, 1)
        # 输出(10,batchsize,512)
        x, _ = self.lstm(x)
        # [bz,10,512]
        x = x.permute(1, 0, 2)
        # 展开
        x = x.reshape(x.shape[0],-1)

        x = self.drop(x)
        y1 = self.fc1(x)
        y2 = self.fc2(x)
        y3 = self.fc3(x)
        y4 = self.fc4(x)
        return y1, y2, y3, y4

    def save(self, circle):
        name = "./weights/EfficientNet_B0_CRNN" + str(circle) + ".pth"
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
    inputs = torch.rand(32, 3, 128, 320)
    model = EfficientNet.from_pretrained('efficientnet-b0',weights_path='./Pretrain_model/efficientnet-b0-355c32eb.pth')
    #print(model)
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
    )
    lstm = nn.LSTM(input_size=320 * 4, hidden_size=512, num_layers=2, bidirectional=True)

    out2 = model1(inputs)  # [32,320,4,10]
    print(out2.shape)
    # (batchsize,320,4,10)->(batchsize,1280,10)
    out2 = out2.reshape(out2.shape[0], -1, out2.shape[-1])
    print(out2.shape)  # [32,1280,10]
    # 维度换位 (10,batchsize,1280)
    out2 = out2.permute(2, 0, 1)  # [10,32,1280]
    print(out2.shape)
    out2 = lstm(out2)

    print(out2.shape())
    # # ----------------- 维度变换 -------------------
    # # torch.squeeze() 维度压缩,将其中size为1的维度都删除
    # # out2 = torch.squeeze(out2)
    # # 维度变换
    # out2 = out2.view(out2.shape[0],-1)
    # print(out2.shape)
    # # -------------------------------------------
    # out2 = nn.Dropout(0.5)(out2)
    # out2 = nn.Linear(in_features=out2.shape[1], out_features=62)(out2)
    # print(out2.shape)
    #
    # # layer = model.
    # feature = model._fc.in_features  # 通道数1280
    # print(feature)
    # model._fc = nn.Linear(in_features=feature,out_features=45)
    # #print(model)
    # # net = timm.create_model('tf_efficientnet_b0_ns',pretrained=True)
    # outputs = model(inputs)
    # # for i in range(len(outputs)):
    # #     print('feature {} shape: {}'.format(i, outputs[i].shape))
    # model.eval()
    #
    # print(outputs.shape)














