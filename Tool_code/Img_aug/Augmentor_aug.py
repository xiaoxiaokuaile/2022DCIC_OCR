import Augmentor
import os


def get_distortion_pipline(path, num):
    # 设置路径
    p = Augmentor.Pipeline(path)
    # 放大缩小
    p.zoom(probability=0.5, min_factor=1.05, max_factor=1.05)
    # 扭曲
    p.random_distortion(probability=1, grid_width=6, grid_height=2, magnitude=3)
    p.sample(num)
    return p


def get_skew_tilt_pipline(path, num):
    p = Augmentor.Pipeline(path)
    # p.zoom(probability=0.5, min_factor=1.05, max_factor=1.05)
    # p.random_distortion(probability=1, grid_width=6, grid_height=2, magnitude=3)
    p.skew_tilt(probability=0.5,magnitude=0.02)
    p.skew_left_right(probability=0.5,magnitude=0.02)
    p.skew_top_bottom(probability=0.5, magnitude=0.02)
    p.skew_corner(probability=0.5, magnitude=0.02)
    p.sample(num)
    return p


def get_rotate_pipline(path, num):
    p = Augmentor.Pipeline(path)
    # p.zoom(probability=0.5, min_factor=1.05, max_factor=1.05)
    # p.random_distortion(probability=1, grid_width=6, grid_height=2, magnitude=3)
    p.rotate(probability=1,max_left_rotation=1,max_right_rotation=1)
    p.sample(num)
    return p


if __name__ == "__main__":
    path = r"./fig"
    # 生成图片数目
    num = 10
    # 创建数据增强实例
    p = Augmentor.Pipeline(path)
    # ------------------------ 放大缩小 -------------------------
    # 放大缩小图像并保持其原有大小
    # min_factor最小倍数, max_factor最大倍数
    p.zoom(probability=0.5, min_factor=1.05, max_factor=1.05)
    # 在图像中的任意位置放大为图像, percentage_area要裁剪的区域占比
    # p.zoom_random(probability=1, percentage_area=0.9, randomise_percentage_area=False)
    # ------------------------- 旋转 ----------------------------
    # 图像旋转
    # max_left_rotation 向左最大旋转角度10,max_right_rotation 向右最大旋转角度10
    # p.rotate(probability=1, max_left_rotation=10, max_right_rotation=10)
    # 图像旋转而不自动裁剪
    # 控制是否应该增加图像的大小以适应旋转。默认值为false，以便图像在旋转后保持原来的尺寸。
    # p.rotate_without_crop(probability=1, max_left_rotation=10, max_right_rotation=10, expand=True)
    # ------------------------ 弹性扭曲 -------------------------
    # 对图像执行随机、弹性变形。
    # grid_width网格中水平轴上矩形个数,grid_height网格中纵轴矩形个数,magnitude扭曲的程度
    # p.random_distortion(probability=1, grid_width=6, grid_height=3, magnitude=3)
    # 对图像执行随机、弹性高斯失真。
    # p.gaussian_distortion(probability=1, grid_width=6, grid_height=3, magnitude=3,
    #                       corner='bell', method='in', mex=0.5, mey=0.5, sdx=0.05, sdy=0.05)
    # ------------------------ 裁剪操作 -------------------------
    # 裁剪出对应size的切片, size需要小于原始size
    # p.crop_by_size(probability=1, width=120, height=50, centre=False)
    # percentage_area要裁剪的取阈占原图百分比
    # p.crop_random(probability=1, percentage_area=0.9, randomise_percentage_area=False)
    # ------------------------ 倾斜扭曲 -------------------------
    # 左右倾斜
    # magnitude:最大倾斜，其值必须介于0.1到1.0之间，其中1表示倾斜45度。
    # p.skew_left_right(probability=1, magnitude=0.1)
    # 前后倾斜
    # p.skew_top_bottom(probability=1, magnitude=0.1)
    # 任意方向倾斜
    # p.skew_tilt(probability=1, magnitude=1)
    # 将图像向一个角落倾斜，随机地按一个随机的大小
    # p.skew_corner(probability=1, magnitude=0.1)
    # ------------------------ 色彩变化 -------------------------
    # 直方图均衡化图像
    # p.histogram_equalisation(probability=1)
    # 图像转换为灰度
    # p.greyscale(probability=1)
    # 图像二值化
    # p.black_and_white(probability=1, threshold=128)
    # 反转图像,色彩反转
    # p.invert(probability=1)
    # 随机改变图像亮度
    # 1表示原图亮度<1表示比原图暗,>1表示比原图亮
    # p.random_brightness(probability=1,min_factor=0.5,max_factor=1.5)
    # 图像饱和度的随机变化
    # p.random_color(probability=1,min_factor=0.5,max_factor=1.5)
    # 随机改变图像对比度
    # p.random_contrast(probability=1,min_factor=0.5,max_factor=1.5)
    # 随机擦除
    # p.random_erasing(probability=1, rectangle_area=0.5)

    p.sample(num)

    # distortion增强
    # p = get_distortion_pipline(path, num)

    # p = get_skew_tilt_pipline(path, num)
    # rotate数据增强
    # p = get_rotate_pipline(path, num)
    # p.process()
