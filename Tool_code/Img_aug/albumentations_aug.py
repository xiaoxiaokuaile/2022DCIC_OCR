# Learner: 王振强
# Learn Time: 2022/2/8 9:52
import cv2
import numpy as np
from albumentations import (Blur, Flip, ShiftScaleRotate, GridDistortion, ElasticTransform, HorizontalFlip, CenterCrop,
                            HueSaturationValue, Transpose, RandomBrightnessContrast, Rotate, CLAHE, RandomCrop, Cutout,Crop,
                            CoarseDropout,CoarseDropout, Normalize, ToFloat, OneOf, Compose, Resize, RandomRain,RandomGridShuffle,
                            PadIfNeeded,RGBShift,RandomBrightness,RandomContrast,GaussianBlur,ChannelShuffle,InvertImg,IAAPerspective,
                            OpticalDistortion,MedianBlur,MotionBlur,GaussNoise,GridDropout,RandomSnow,RandomRain,IAASharpen,IAAAdditiveGaussianNoise,
                            RandomFog, Lambda, ChannelDropout, ISONoise, VerticalFlip, RandomGamma, RandomRotate90)
import matplotlib.pyplot as plt

"""
    albumentations 数据增强工具的使用
"""

if __name__ == '__main__':
    image = 'fig/46.png'
    image = cv2.imread(image)
    # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # image = image.astype(np.uint8)

    # 单独变换
    image2 = RandomRotate90(p=1)(image=image)['image'] # 随机旋转90°

    # 组合变换
    image3 = Compose([
        # 修改图片尺寸
        Resize(40,100,p=1),
        # ----------------------------------------------------------------------
        # 并行操作中随机选择一个
        OneOf([
            RandomGamma(gamma_limit=(60, 120), p=1),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
        ],p=0),
        # ----------------------------------------------------------------------
        # Blur 模糊图像 # 使用随机大小的内核模糊输入图像
        # Blur(blur_limit=5, always_apply=False, p=1),
        # 高斯滤波
        # GaussianBlur(blur_limit=7, always_apply=False, p=1),
        # 运动模糊,给图像加上运动模糊。运动模糊是景物图象中的移动效果。
        # MotionBlur(blur_limit=7, always_apply=False, p=1),
        # 中心模糊, 图像中值滤波
        # MedianBlur(blur_limit=7, always_apply=False, p=1),
        # 锐化
        # IAASharpen(p=1),
        # 围绕X轴垂直翻转输入 (横轴)
        # VerticalFlip(always_apply=False, p=1),
        # 围绕y轴水平翻转输入 (纵轴)
        # HorizontalFlip(always_apply=False, p=1),
        # 水平，垂直或水平和垂直翻转输入。
        # Flip(always_apply=False, p=1),
        # 转置
        # Transpose(always_apply=False, p=1),
        # 裁剪图像，裁剪出 (xmin,ymin) 和 (xmax,ymax) 之间的区域
        # Crop(x_min=0, y_min=0, x_max=80, y_max=30, always_apply=False, p=1.0),
        # 透视变换
        # IAAPerspective(p=1),
        # 裁剪输入的中心部分。
        # CenterCrop(30, 80, always_apply=False, p=1.0),
        # 裁剪输入的随机部分。
        # RandomCrop(30, 80, always_apply=False, p=1.0),
        # 随机Gamma变换(看不出来效果)
        # RandomGamma(gamma_limit=(80, 120), eps=None, always_apply=False, p=1),
        # 随机应用仿射变换: 平移 缩放 旋转 三合一
        # rotate_limit 图片旋转范围[-5,5], shift_limit 图片宽高的平移因子[-0.0625,0.0625]
        # scale_limit 图片缩放因子[-0.1,0.1], border_mode 用于指定外插算法, p 转换的概率
        ShiftScaleRotate(rotate_limit=5, shift_limit=0.0625, scale_limit=0.1, p=0.5,border_mode=cv2.BORDER_REPLICATE),
        # Rotate旋转 随机旋转图片(默认使用reflect方法扩充图片，可以改为参数等其他方法填充)。
        # Rotate(limit=10, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=1),
        # 网格失真(明显)
        # GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, border_mode=4,
        #                value=None, mask_value=None,always_apply=False, p=1),
        # 弹性变换(夸张)
        # ElasticTransform(alpha=1, sigma=50, alpha_affine=50, interpolation=1, border_mode=4,
        #                  value=None,mask_value=None, always_apply=False, approximate=False, p=1),
        # 随机网格洗牌
        # RandomGridShuffle(grid=(3, 3), always_apply=False, p=1)
        # 参数：随机色调、饱和度、值变化。
        HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5)
        # 填充图像。 padding
        # PadIfNeeded(min_height=50, min_width=120, border_mode=4, value=None, mask_value=None,
        #             always_apply=False,p=1.0)
        # Padding
        # PadIfNeeded(min_height=60, min_width=120, border_mode=4, value=None, mask_value=None,
        #             always_apply=False, p=1.0),
        # 参数：随机平移R、G、B通道值。(有用)
        # RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=1)
        # 随机重新排列RGB图像通道
        # ChannelShuffle(always_apply=False, p=1),
        # 随机亮度变化。
        # RandomBrightness(limit=0.2, always_apply=False, p=1),
        # 随机对比度变化。
        # RandomContrast(limit=0.2, always_apply=False, p=1),
        # 反转图像
        # InvertImg(always_apply=False, p=1),
        # 随机擦除
        # Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=1),
        # 雾化
        # RandomFog(fog_coef_lower=0.3, fog_coef_upper=1, alpha_coef=0.08, always_apply=False, p=1),
        # 加雪花
        # RandomSnow(p=1),
        # 加雨滴
        # RandomRain(p=1),
        # 光学畸变
        # OpticalDistortion(distort_limit=0.05, shift_limit=0.05, interpolation=1, border_mode=4, value=None,
        #                   mask_value=None, always_apply=False, p=1),
        # GaussNoise 高斯噪声
        # GaussNoise(var_limit=(10.0, 50.0),always_apply=False, p=1),
        # 对比度受限的自适应直方图均衡化
        # CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=1),
        # 在图像中生成正方形区域。
        # Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=1),
        # 在图像上生成矩形区域
        # CoarseDropout(max_holes=8, max_height=8, max_width=8, min_height=None, min_width=None,
        #                fill_value=0, always_apply=False, p=1),



    # ----------------------------------------------------------------------
        # 图像归一化
        # Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
        ],p=1)(image=image)["image"]

    image4 = Compose([
        Resize(height=40, width=100, p=1),
        # 噪声
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.3),
        # 模糊,锐化,滤波
        OneOf([
            Blur(blur_limit=5, always_apply=False),
            # 雾化
            RandomFog(fog_coef_lower=0.3, fog_coef_upper=1, alpha_coef=0.08, always_apply=False),
            # 加雪花
            RandomSnow(),
            # 加雨滴
            RandomRain(),
            # 锐化
            IAASharpen(),
        ],p=0.5),
        OneOf([
            # 平移, 缩放和旋转
            ShiftScaleRotate(rotate_limit=5, shift_limit=0.0625, scale_limit=0.1,border_mode=cv2.BORDER_REPLICATE),
            # 透视变换
            IAAPerspective(),
        ],p=0.5),

        # 颜色增强方案:
        OneOf([
            # 参数：随机平移R、G、B通道值。(有用)
            RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False),
            # 随机重新排列RGB图像通道
            ChannelShuffle(always_apply=False),
            # 随机亮度变化。
            RandomBrightness(limit=0.2, always_apply=False),
            # 随机对比度变化。
            RandomContrast(limit=0.2, always_apply=False),
            #  随机更改输入图像的色相,饱和度和值
            HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10),
        ],0.5),
        # 归一化
        # Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0),
    ])(image=image)["image"]

    cv2.imshow("www",image)
    cv2.imshow("aug img",image3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


























