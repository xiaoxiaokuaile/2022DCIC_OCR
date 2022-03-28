# Learner: 王振强
# Learn Time: 2022/2/15 12:55
import numpy as np
import pandas as pd
import os, time, glob
import cv2
from sklearn.model_selection import KFold
import argparse
import csv

"""
    1.图片分类方法切分N折交叉
    2.制作伪标签数据集
"""


# 制作 训练集 + 测试集 数据CSV表格
def make_data_csv(root, save_csv_path, Istrain=True):
    if Istrain:
        images = glob.glob(os.path.join(root, '*.png'))
        # Linux
        # labels = [image.split('.')[-2].split('/')[-1] for image in images]
        # windows
        img_name = [image.split('\\')[-1] for image in images]
        labels = [image.split('.')[-2].split('\\')[-1] for image in images]
        train = pd.DataFrame(index=range(len(images)))
        train['ImgName'] = img_name
        train['label'] = labels
        train['fold'] = -1
        train[['ImgName', 'label','fold']].to_csv(os.path.join(save_csv_path,'Pseudo.csv'), index=None)
    else:
        fn = sorted(glob.glob(os.path.join(root, '*.png')),key=lambda x: int(x.split('.')[-2].split('\\')[-1]))
        img_name = [image.split('\\')[-1] for image in fn]
        label = ["-" for i in range(len(fn))]
        r = pd.DataFrame(data={"ImgName": img_name, "label": label})
        r.to_csv(os.path.join(save_csv_path,'test.csv'), index=False)


# 通过confidence文件制作伪标签数据集
def make_Pseudo_tag_dataset(df_confidence):
    # 遍历测试集csv
    for fold_id, index in enumerate(df_confidence.index):
        img_name = df_confidence.iloc[index]['num']
        label = df_confidence.iloc[index]['tag']
        confidence = df_confidence.iloc[index]['confidence']

        # 可以解决中文路径无法读取的问题
        src_image = cv2.imdecode(np.fromfile('./test/'+ str(img_name) + '.png', dtype=np.uint8), -1)

        if confidence > 0.999:
            # print(img_name,label,confidence)
            # 重命名保存切片
            cv2.imwrite('./Pseudo/' + label + '.png', src_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 原始训练集图片路径
    parser.add_argument('--root_train_path', type=str, default=r'./Pseudo/')
    # 测试集图片路径
    parser.add_argument('--root_test_path', type=str, default=r'./test/')
    # 训练集CSV保存路径
    parser.add_argument('--save_data_CSV', type=str,default='./')
    opt = parser.parse_args()

    root_train = opt.root_train_path
    root_test_path = opt.root_test_path
    save_data_CSV = opt.save_data_CSV

    # -----------------------------------------------------------
    # 制作训练集CSV
    # save_train_CSV = './'
    # make_data_csv(root_train, save_train_CSV, Istrain=True)
    # 制作测试集CSV
    # make_data_csv(root_test_path, save_data_CSV, Istrain=False)
    # ------------------------------------------------------------
    # -------------------- 五折交叉标记 ----------------------
    df_train = pd.read_csv('train.csv')
    df_train['fold'] = -1
    N_FOLDS = 5
    strat_kfold = KFold(n_splits=N_FOLDS, random_state=2022, shuffle=True)
    for fold_id, (train_index, val_index) in enumerate(strat_kfold.split(df_train.index)):
        # 分割训练集验证集
        X_train = df_train.iloc[train_index]
        X_val = df_train.iloc[val_index]


        df_train.iloc[val_index, -1] = fold_id
        df_train['fold'] = df_train['fold'].astype('int')
        df_train.to_csv('./train_fold.csv', index=None)
    # -------------------------------------------------------

    # ------------------- 制作伪标签数据集 ----------------------
    # # num,tag,confidence
    # df_confidence = pd.read_csv('./Csv_Path/confidence_all.csv')
    # make_Pseudo_tag_dataset(df_confidence)
    # --------------------------------------------------------
    # ----------- 合并Pseudo.csv + train_fold.csv ------------
    # Pseudo_df = pd.read_csv('Pseudo.csv')
    # Train_fold_df = pd.read_csv('train_fold.csv')
    # # print(Pseudo_df[Pseudo_df.ImgName =='00IS.png'])
    # for fold_id, index in enumerate(Train_fold_df.index):
    #     img_name = Train_fold_df.iloc[index]['ImgName']
    #     label = Train_fold_df.iloc[index]['label']
    #     fold = Train_fold_df.iloc[index]['fold']
    #     Pseudo_df.loc[Pseudo_df.ImgName == img_name,'fold'] = fold
    # Pseudo_df[['ImgName', 'label','fold']].to_csv(('Pseudo_fold.csv'), index=None)
    # -------------------------------------------------------

















