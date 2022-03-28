# Learner: 王振强
# Learn Time: 2022/2/16 16:35
import torch
from torch import nn
import os
from .repvgg import create_RepVGG_B2g4


class RepVGG_B2g4(nn.Module):
    def __init__(self, class_num=62):
        super(RepVGG_B2g4, self).__init__()
        pr_model = create_RepVGG_B2g4(deploy=False)
        pr_model.load_state_dict(torch.load('./model/RepVGG/RepVGG-B2g4-train.pth'))
        self.cnn = nn.Sequential(
            pr_model.stage0,
            pr_model.stage1,
            pr_model.stage2,
            pr_model.stage3,
            pr_model.stage4,
            pr_model.gap,
        )
        # 将特征图转化为(1,1)大小
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(0.2)
        self.fc1 = nn.Linear(2560, class_num)
        self.fc2 = nn.Linear(2560, class_num)
        self.fc3 = nn.Linear(2560, class_num)
        self.fc4 = nn.Linear(2560, class_num)

    def forward(self, x):
        x = self.cnn(x)
        # x = self.avgpool(x)
        # 维度变换
        x = x.view(x.shape[0], -1)
        x = self.drop(x)
        y1 = self.fc1(x)
        y2 = self.fc2(x)
        y3 = self.fc3(x)
        y4 = self.fc4(x)
        return y1, y2, y3, y4

    def save(self, circle):
        name = "./weights/RepVGG_B2g4" + str(circle) + ".pth"
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
    inputs = torch.rand(32, 3, 224, 224)
    model = create_RepVGG_B2g4(deploy=False)
    model.load_state_dict(torch.load('./RepVGG/RepVGG-B2g4-train.pth'))
    print(model)
    model1 = nn.Sequential(
        model.stage0,   # 64 (2,2) (224,224)->(112,112)
        model.stage1,   # 160 (2,2) (112,112)->(56,56)
        model.stage2,   # 320 (2,2) (56,56)->(28,28)
        model.stage3,   # 640 (2,2) (28,28)->(14,14)
        model.stage4,   # 2560 (2,2) (14,14)->(7,7)
        model.gap,   # 2560 (2,2) (14,14)->(7,7)
    )
    out2 = model1(inputs)
    print(out2.shape)















