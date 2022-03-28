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
                   EfficientNet_B5,EfficientNet_B6,EfficientNet_B7,EfficientNet_B8,
                   EfficientNet_B0_CRNN,EfficientNet_B0_sig)
from model import (RepVGG_A0,RepVGG_A1,RepVGG_A2, RepVGG_B0,RepVGG_B1,RepVGG_B1g2,RepVGG_B1g4,
                   RepVGG_B2,RepVGG_B2g4,RepVGG_B3,RepVGG_B3g4,RepVGG_D2se)

# 使用显卡,一张显卡就是0
os.environ['CUDA_VISIBLE_DEVICES']='0'


def predict(model, dataLoader, csv_file,confidence_file):
    # --------------------- 打开需要保存的CSV文件 ----------------------
    f = open(csv_file,"w",encoding='utf-8',newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["num","tag"])
    # -------------------- 打开保存置信度的CSV文件 ----------------------
    c = open(confidence_file, "w",encoding='utf-8',newline='')
    csv_confidence = csv.writer(c)
    csv_confidence.writerow(["num", "tag","confidence"])
    # ---------------------------------------------------------------
    # ------------------------- 遍历测试集 ---------------------------
    for circle, input in enumerate(dataLoader, 0):
        x, label = input
        # 获取图片名称不带后缀
        label = list(label)[0].split('.')[0]
        if torch.cuda.is_available():
            x = x.cuda()   # 将x拿到显卡上运算
        # ---------------------- 模型预测结果 ------------------------
        y1, y2, y3, y4 = model(x)  # torch.Size([1, 62])
        # softmax激活
        y1, y2, y3, y4 = F.softmax(y1,dim=1),F.softmax(y2,dim=1),F.softmax(y3,dim=1),F.softmax(y4,dim=1)
        idx1,idx2,idx3,idx4 = torch.argmax(y1),torch.argmax(y2),torch.argmax(y3),torch.argmax(y4)
        # 获取confidence
        # confidence = (y1[0][idx1] + y2[0][idx2] + y3[0][idx3] + y4[0][idx4])/4
        confidence = min(y1[0][idx1], y2[0][idx2], y3[0][idx3], y4[0][idx4])

        # y.topk(k,dim) 沿dim维度返回输入张量y中k个最大值, 返回(value,index)元组, 需要Index转换为字符
        y1, y2, y3, y4 = y1.topk(1, dim=1)[1].view(1, 1),y2.topk(1, dim=1)[1].view(1, 1),y3.topk(1, dim=1)[1].view(1, 1),y4.topk(1, dim=1)[1].view(1, 1)
        y = torch.cat((y1, y2, y3, y4), dim=1)  # (1,4)大小数组
        decLabel = LabeltoStr([y[0][0], y[0][1], y[0][2], y[0][3]])
        # ----------------------------------------------------------
        csv_writer.writerow([label,decLabel])
        # 保存含有confidence的结果
        csv_confidence.writerow([label, decLabel,confidence.cpu().detach().numpy()])
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

        y_1, y_2, y_3, y_4 = torch.zeros(1, 62), torch.zeros(1, 62), torch.zeros(1, 62), torch.zeros(1, 62)
        confidence_ad = torch.zeros(1)
        if torch.cuda.is_available():
            # 下列tensor都放到显卡上运算
            x = x.cuda()
            y_1, y_2, y_3, y_4 = y_1.cuda(), y_2.cuda(), y_3.cuda(), y_4.cuda()
            confidence_ad = confidence_ad.cuda()

        # 遍历模型列表预测结果
        for i in range(num_of_model):
            # 得到对应模型数值输出
            y1, y2, y3, y4 = model_list[i](x)  # torch.Size([1, 62])  有正有负
            # softmax激活
            y1, y2, y3, y4 = F.softmax(y1, dim=1), F.softmax(y2, dim=1), F.softmax(y3, dim=1), F.softmax(y4, dim=1)
            idx1, idx2, idx3, idx4 = torch.argmax(y1), torch.argmax(y2), torch.argmax(y3), torch.argmax(y4)
            # 获取confidence
            # confidence = (y1[0][idx1] + y2[0][idx2] + y3[0][idx3] + y4[0][idx4])/4
            confidence = min(y1[0][idx1], y2[0][idx2], y3[0][idx3], y4[0][idx4])
            confidence_ad = confidence_ad + confidence / num_of_model

            y_1 = y_1 + y1 / num_of_model
            y_2 = y_2 + y2 / num_of_model
            y_3 = y_3 + y3 / num_of_model
            y_4 = y_4 + y4 / num_of_model

        # 模型融合后结果
        y1_out, y2_out, y3_out, y4_out = y_1.topk(1, dim=1)[1].view(1, 1), \
                                        y_2.topk(1, dim=1)[1].view(1, 1), \
                                        y_3.topk(1, dim=1)[1].view(1, 1), \
                                        y_4.topk(1, dim=1)[1].view(1, 1)
        y = torch.cat((y1_out, y2_out, y3_out, y4_out), dim=1)  # (1,4)大小数组
        decLabel = LabeltoStr([y[0][0], y[0][1], y[0][2], y[0][3]])
        # 保存模型融合后的结果
        csv_writer.writerow([label, decLabel])
        # 保存含有confidence的结果
        csv_confidence.writerow([label, decLabel,confidence_ad.cpu().detach().numpy()])
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
    userTestDataLoader = DataLoader(testDataset, batch_size=1,shuffle=False, num_workers=1)

    # -------------- 单模型预测 -------------
    model_pr = EfficientNet_B5()
    # 模型预测
    model_pr.eval()
    model_pr.load_model("./weights/EfficientNet_B3_Flod_1.pth")
    if torch.cuda.is_available():
        model_pr = model_pr.cuda()

    predict(model_pr, userTestDataLoader, csv_file="./submission.csv",confidence_file="./confidence_all.csv")
    # ------------------------------------
    # ------------ 多模型融合 --------------
    # model1 = EfficientNet_B3()
    # model2 = EfficientNet_B3()
    # model3 = EfficientNet_B3()
    # model4 = EfficientNet_B3()
    # model1.eval()
    # model2.eval()
    # model3.eval()
    # model4.eval()
    # # 模型加载
    # model1.load_model("./weights/EfficientNet_B3_Flod_1.pth")
    # model2.load_model("./weights/EfficientNet_B3_Flod_2.pth")
    # model3.load_model("./weights/EfficientNet_B3_Flod_3.pth")
    # model4.load_model("./weights/EfficientNet_B3_Flod_4.pth")
    #
    # if torch.cuda.is_available():
    #     model1 = model1.cuda()
    #     model2 = model2.cuda()
    #     model3 = model3.cuda()
    #     model4 = model4.cuda()
    # # 模型列表
    # model_list = []
    # model_list.append(model1)
    # model_list.append(model2)
    # model_list.append(model3)
    # model_list.append(model4)
    # 多模型融合
    # predict_all(model_list, userTestDataLoader, csv_file="./submission_all.csv", confidence_file="./confidence_all.csv")

