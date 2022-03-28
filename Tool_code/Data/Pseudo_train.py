# Learner: 王振强
# Learn Time: 2022/3/15 9:23
import numpy as np
import pandas as pd
import os, time, glob
import cv2
from sklearn.model_selection import KFold
import csv


# 数据增强的数据重命名
def aug_data(root,savepath):
    # 获取图片列表
    images = glob.glob(os.path.join(root, '*.png'))
    for index,image in enumerate(images):
        # 可以解决中文路径无法读取的问题
        src_image = cv2.imdecode(np.fromfile(image, dtype=np.uint8), -1)
        label = image.split('.')[-3].split('\\')[-1].split('_')[2]
        # label = image.split('.')[-2].split('\\')[-1]
        font = image.split('.')[-3].split('\\')[-1].split('_')[4]
        fold = image.split('.')[-3].split('\\')[-1].split('_')[6]
        # print(label)
        img_name = label + '_font_' + font + '_fold_' + fold + '_aug_' + str(index) + '.png'
        # print(img_name)
        # # 保存图片
        cv2.imwrite(os.path.join(savepath,img_name), src_image)



# 测试模型准确率
def make_csv(font_csv, sub_confidence_csv, img_last_path, csv_file):
    # 打开font文件
    df_font = pd.read_csv(font_csv)
    # 打开结果文件
    df_confidence = pd.read_csv(sub_confidence_csv)
    # 获取训练集图片列表
    images = glob.glob(os.path.join(img_last_path, '*.png'))
    # 获取图片索引
    nums_list = [image.split('.')[-2].split('\\')[-1] for image in images]
    # --------------------- 打开需要保存的CSV文件 ----------------------
    f = open(csv_file, "w", encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["num", "tag"])
    # 计数
    sum = 0
    # 遍历csv
    for fold_id, index in enumerate(df_confidence.index):
        img_name = df_confidence.iloc[index]['num']
        label = df_confidence.iloc[index]['tag']
        confidence = df_confidence.iloc[index]['confidence']
        font = df_font.iloc[index]['font']
        if index < 10000:
            if confidence > 0:  # 直接筛选
            # if confidence > 0 and str(img_name) in nums_list:    # 结合没有训练数据筛选
                if font == 3:
                    csv_writer.writerow([img_name, label])
                    sum += 1
                else:
                    # pass
                    csv_writer.writerow([img_name, '0000'])
            else:
                # pass
                csv_writer.writerow([img_name, '0000'])
        else:
            # pass
            csv_writer.writerow([img_name, '0000'])
    print(sum)


# 进一步筛选伪标签
def make_pseudo_dataset(sub_confidence_csv,font_csv,srcimg_path,img_last_path,save_img_path,save_last_path):
    # 打开font文件
    df_font = pd.read_csv(font_csv)
    # 打开结果文件
    df_confidence = pd.read_csv(sub_confidence_csv)
    # 获取训练集图片列表
    images = glob.glob(os.path.join(img_last_path, '*.png'))
    # 获取图片索引
    nums_list = [image.split('.')[-2].split('\\')[-1] for image in images]
    # 计数
    sum = 0
    # 遍历csv
    for fold_id, index in enumerate(df_confidence.index):
        img_name = df_confidence.iloc[index]['num']
        label = df_confidence.iloc[index]['tag']
        confidence = df_confidence.iloc[index]['confidence']
        font = df_font.iloc[index]['font']

        # 可以解决中文路径无法读取的问题
        src_image = cv2.imdecode(np.fromfile(os.path.join(srcimg_path, str(img_name) + '.png'), dtype=np.uint8), -1)
        if index < 100000:
            if font == 3:
                # if confidence > 0.997:
                if confidence > 0.999 and str(img_name) in nums_list:
                    # 重命名保存图片
                    cv2.imwrite(os.path.join(save_img_path, label + '.png'), src_image)
                    sum += 1
                elif str(img_name) in nums_list:
                    cv2.imwrite(os.path.join(save_last_path, str(img_name) + '.png'), src_image)
                else:
                    pass
            else:
                pass
        else:
            pass
    print(sum)


# 根据图片文件夹制作五折交叉csv文档
def K_flod_dataset(root, save_csv_path):
    # 获取训练集图片列表
    images = glob.glob(os.path.join(root, '*.png'))
    img_name = [image.split('\\')[-1] for image in images]
    # 获取标签
    labels = [image.split('.')[-2].split('\\')[-1].split('_')[0] for image in images]
    # # 获取字体
    # font = [image.split('.')[-2].split('\\')[-1].split('_')[2] for image in images]
    # # 获取折数
    # fold = [image.split('.')[-2].split('\\')[-1].split('_')[4] for image in images]
    # # 获取数据来源
    # aug = [image.split('.')[-2].split('\\')[-1].split('_')[5] for image in images]

    train = pd.DataFrame(index=range(len(images)))
    train['ImgName'] = img_name
    train['label'] = labels
    # train['font'] = font
    train['fold'] = -1
    # train['aug'] = aug

    N_FOLDS = 5
    strat_kfold = KFold(n_splits=N_FOLDS, random_state=2022, shuffle=True)
    for fold_id, (train_index, val_index) in enumerate(strat_kfold.split(train.index)):
        # fold
        train.iloc[val_index, -1] = fold_id
        # 转换类型
        train['fold'] = train['fold'].astype('int')
    # 保存训练csv文件
    train[['ImgName', 'label','fold']].to_csv(save_csv_path, index=None)


# 将原始数据集五折划分
def font_K_aug(font_csv,src_img,save_path):
    # 打开font csv文件
    df_font = pd.read_csv(font_csv)
    # 遍历csv
    for fold_id, index in enumerate(df_font.index):
        img_name = df_font.iloc[index]['ImgName']
        label = df_font.iloc[index]['label']
        fold = df_font.iloc[index]['fold']
        src_img_path = os.path.join(src_img,img_name)
        save_img_path = os.path.join(os.path.join(save_path,str(fold)),label + '_font_1_fold_' + str(fold) + '_old' + '.png')
        # 可以解决中文路径无法读取的问题
        src_image = cv2.imdecode(np.fromfile(src_img_path, dtype=np.uint8), -1)
        # 重命名保存图片
        cv2.imwrite(save_img_path, src_image)



if __name__ == '__main__':
    # --------------------- 离线数据增强 --------------------------
    root__ = r'./new_training_dataset/font_3_Kfold/4/output'
    save_path__ = r'./new_training_dataset/font_3_Kfold/fold_4_aug'
    # aug_data(root__, save_path__)
    # ------------------- 获取指定图片准确率 ------------------------
    # confidence_csv = './Csv_Path/check_font.csv'
    confidence_csv = './Csv_Path/confidence_all.csv'
    font_csv = './Csv_Path/test_font.csv'  # 测试集字体编号文件
    img_last_path = './train_test_add/font3_last'  # 需要继续筛选的图片路径
    csv_file = './Csv_Path/confidence_font_3.csv'  # 生成结果保存路径
    # make_csv(font_csv,confidence_csv,img_last_path,csv_file)

    # 制作伪标签数据集
    # sub_confidence_csv = './Csv_Path/check_font.csv'
    sub_confidence_csv = './Csv_Path/confidence_all.csv'
    srcimg_path = './old_dataset/test'
    save_img_path = './train_test_add/font_3_pseudo'  # 伪标签保存
    save_last_path = './train_test_add/font3_last'   # 准确率不够的图片保存
    # make_pseudo_dataset(sub_confidence_csv, font_csv, srcimg_path, img_last_path,save_img_path,save_last_path)

    # ----------------- 根据图片文件夹制作五折交叉 ------------------
    root_k = './new_training_dataset/make_15000/data_15000'
    save_csv_path = './new_training_dataset/make_15000/font_make_aug_Kflod.csv'
    K_flod_dataset(root_k, save_csv_path)
    # ----------------------------------------------------------
    font_ = r'./new_training_dataset/font1_all/font_1_all_Kflod.csv'
    src_img = r'./new_training_dataset/font1_all/font_1'
    save_ = r'./new_training_dataset/font_1_Kfold'
    # font_K_aug(font_, src_img, save_)