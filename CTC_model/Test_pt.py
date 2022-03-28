# Learner: 王振强
# Learn Time: 2022/2/3 23:30
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.utils import *
from utils.data_loader import CaptDataset
from train import LitPlants
import pandas as pd
import time
import numpy as np
import os, glob
import torch
from config.config import config
# model
from model import Model_resnet18,Model_resnet101,EfficientNet_B4,EfficientNet_B5,EfficientNet_B0


def testimg2csv():
    fn = sorted(glob.glob(os.path.join(config.testPath,"*.png")),key=lambda x: int(x.split('.')[-2].split('/')[-1]))
    label = ["-" for i in range(len(fn))]
    r = pd.DataFrame(data={"filename": fn, "label": label})
    r.to_csv("./test.csv", index=False)

##################
def predict(weight_file, TTA):
    print(f"filename:{weight_file},tta:{TTA}")

    model = EfficientNet_B0()
    # 模型预测
    model.eval()
    model.load_model(weight_file)
    if torch.cuda.is_available():
        model = model.cuda()

    # 开启TTA时候，注意修改数据增强
    test_csv = pd.read_csv("./data/test.csv")
    testDataset = CaptDataset(test_csv, config.testPath, data_mode='test')

    test_loader = DataLoader(dataset=testDataset,batch_size=32,shuffle=False, num_workers=4)

    ttar = []
    with torch.no_grad():
        for i in range(TTA):
            result = []
            for image, filename in tqdm(test_loader):
                if torch.cuda.is_available():
                    image = image.cuda()
                output = model(image)
                output = decode_predictions(output.cpu())
                for i in range(len(output)):
                    pred = output[i]
                    result.append([filename[i].split('.')[-2].split('/')[-1], pred])
            result = pd.DataFrame(np.array(result), columns=["num", "tag"])
            ttar.append(result)

    r = ttar[0]

    r.to_csv("submit.csv", index=False)


# 多线程
def processing_predict(args):
    return predict(args[0], args[1], args[2])


if __name__ == '__main__':
    # 制作测试集CSV文件
    # testimg2csv()
    # 保存路径
    model_path = "./weights/EfficientNet_B0_best.pth"
    # 预测
    predict(model_path, 1)












