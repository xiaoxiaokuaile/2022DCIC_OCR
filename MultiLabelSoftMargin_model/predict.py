import csv
import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
# 加载数据用函数
from torch.utils.data import DataLoader
from utils.dataset import *
# model
from model import (EfficientNet_B0,EfficientNet_B1,EfficientNet_B2,EfficientNet_B3,EfficientNet_B4,
                   EfficientNet_B5,EfficientNet_B6,EfficientNet_B7,EfficientNet_B8)
from model import (RepVGG_A0,RepVGG_A1,RepVGG_A2, RepVGG_B0,RepVGG_B1,RepVGG_B1g2,RepVGG_B1g4,
                   RepVGG_B2,RepVGG_B2g4,RepVGG_B3,RepVGG_B3g4,RepVGG_D2se)

# 使用显卡,一张显卡就是0
os.environ['CUDA_VISIBLE_DEVICES']='0'


source = [str(i) for i in range(0, 10)]
source += [chr(i) for i in range(97, 97 + 26)]
source += [chr(i) for i in range(65, 65 + 26)]
alphabet = ''.join(source)


def predict(model, dataLoader, csv_file,confidence_file):
    # --------------------- 打开需要保存的CSV文件 ----------------------
    f = open(csv_file,"w",encoding='utf-8',newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["num","tag"])
    # -------------------- 打开保存置信度的CSV文件 ---------------------
    c = open(confidence_file, "w",encoding='utf-8',newline='')
    csv_confidence = csv.writer(c)
    csv_confidence.writerow(["num", "tag","confidence"])
    # ------------------------- 遍历测试集 ---------------------------
    for circle, input in enumerate(dataLoader, 0):
        x, label = input
        # 获取图片名称不带后缀
        label = list(label)[0].split('.')[0]
        if torch.cuda.is_available():
            x = x.cuda()   # 将x拿到显卡上运算
        # ---------------------- 模型预测结果 ------------------------
        output = model(x)
        output = output.view(-1, 62) # 将 [248]->[4,62]
        output_softmax = F.softmax(output, dim=1)  # [4,62]
        output_argmax = torch.argmax(output_softmax, dim=1)
        output_idx = output_argmax.view(-1, 4)[0]
         # 获取confidence
        confidence = min(output_softmax[0][output_idx[0]], \
                         output_softmax[1][output_idx[1]], \
                         output_softmax[2][output_idx[2]], \
                         output_softmax[3][output_idx[3]])

        decLabel = ''.join([alphabet[i] for i in output_idx.cpu().numpy()])
        # ----------------------------------------------------------
        csv_writer.writerow([label,decLabel])
        # 保存含有confidence的结果
        csv_confidence.writerow([label, decLabel,float(confidence.cpu().detach())])
        if circle % 100 == 0:
            print("%d\t%-9s\t%-4s\t%.4f" % (circle, label, decLabel,confidence))
    f.close()


# 模型融合
def predict_all(model_list, dataLoader, csv_file,confidence_file):
    # --------------------- 打开需要保存的CSV文件 ----------------------
    f = open(csv_file, "w",encoding='utf-8',newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["num", "tag"])
    # -------------------- 打开保存置信度的CSV文件 ----------------------
    c = open(confidence_file, "w",encoding='utf-8',newline='')
    csv_confidence = csv.writer(c)
    csv_confidence.writerow(["num", "tag", "confidence"])
    # ---------------------------------------------------------------
    # 模型个数
    num_of_model = len(model_list)
    # 遍历测试集
    for circle, input in enumerate(dataLoader, 0):
        x, label = input
        label = list(label)[0].split('.')[0]

        y_add = torch.zeros(4, 62)
        confidence_ad = torch.zeros(1)
        if torch.cuda.is_available():
            # 下列tensor都放到显卡上运算
            x = x.cuda()
            y_add = y_add.cuda()
            confidence_ad = confidence_ad.cuda()

        # 遍历模型列表预测结果
        for i in range(num_of_model):
            # 得到对应模型数值输出
            output = model_list[i](x)  # 248
            output = output.view(-1, 62)
            output_softmax = F.softmax(output, dim=1)  # [4,62]
            output_argmax = torch.argmax(output_softmax, dim=1)
            output_idx = output_argmax.view(-1, 4)[0]
            # 获取confidence
            confidence = min(output_softmax[0][output_idx[0]], \
                             output_softmax[1][output_idx[1]], \
                             output_softmax[2][output_idx[2]], \
                             output_softmax[3][output_idx[3]])
            confidence_ad = confidence_ad + confidence / num_of_model
            y_add = y_add + output_softmax / num_of_model

        y_argmax = torch.argmax(y_add, dim=1)
        y_idx = y_argmax.view(-1, 4)[0]
        decLabel = ''.join([alphabet[i] for i in y_idx.cpu().numpy()])

        # 保存模型融合后的结果
        csv_writer.writerow([label, decLabel])
        # 保存含有confidence的结果
        csv_confidence.writerow([label, decLabel,float(confidence_ad.cpu().detach())])
        if circle % 100 == 0:
            print("%d\t%-9s\t%-4s\t%.4f" % (circle, label, decLabel,confidence_ad))

    f.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="weightpath")
    # 测试集图片路径
    parser.add_argument("--test_path", type=str, default="./data/test/")
    parser.add_argument("--test_csv_path", type=str, default="./data/test.csv")
    opt = parser.parse_args()

    # 读取测试集数据
    test_csv = pd.read_csv(opt.test_csv_path)
    testDataset = Captcha(test_csv, opt.test_path, data_mode='test')
    userTestDataLoader = DataLoader(testDataset, batch_size=1,shuffle=False, num_workers=4)

    # -------------- 单模型预测 -------------
    # model_pr = EfficientNet_B0()
    # # 模型预测
    # model_pr.eval()
    # model_pr.load_model("./weights/EfficientNet_B0_best.pth")
    # if torch.cuda.is_available():
    #     model_pr = model_pr.cuda()
    #
    # predict(model_pr, userTestDataLoader, csv_file="./submission.csv",confidence_file="./confidence.csv")
    # ------------------------------------
    # ------------ 多模型融合 --------------
    model1 = EfficientNet_B0()
    model2 = EfficientNet_B1()
    # model3 = EfficientNet_B6()
    # model4 = EfficientNet_B6()
    model1.eval()
    model2.eval()
    # model3.eval()
    # model4.eval()
    # # 模型加载
    model1.load_model("./weights/EfficientNet_B0_best.pth")
    model2.load_model("./weights/EfficientNet_B1_best.pth")
    # model3.load_model("./weights/EfficientNet_B6_Flod_3.pth")
    # model4.load_model("./weights/EfficientNet_B6_Flod_4.pth")
    #
    if torch.cuda.is_available():
        model1 = model1.cuda()
        model2 = model2.cuda()
    #     model3 = model3.cuda()
    #     model4 = model4.cuda()
    # # 模型列表
    model_list = []
    model_list.append(model1)
    model_list.append(model2)
    # model_list.append(model3)
    # model_list.append(model4)
    # 多模型融合
    predict_all(model_list, userTestDataLoader, csv_file="./submission_all.csv", confidence_file="./confidence_all.csv")

