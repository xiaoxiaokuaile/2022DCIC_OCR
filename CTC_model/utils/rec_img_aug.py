import math
import cv2
import numpy as np
import random
from .text_image_aug import tia_perspective, tia_stretch, tia_distort
import torch
from math import floor, ceil
from PIL import Image, ImageOps, ImageEnhance


# ----------------------------- 数据增强 ----------------------------
def flag():
    """
    flag
    """
    return 1 if random.random() > 0.5000001 else -1


# 颜色空间转换
def cvtColor(img):
    """
    cvtColor
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    delta = 0.001 * random.random() * flag()
    hsv[:, :, 2] = hsv[:, :, 2] * (1 + delta)
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return new_img


# 模糊
def blur(img):
    """
    blur
    """
    h, w, _ = img.shape
    if h > 10 and w > 10:
        return cv2.GaussianBlur(img, (5, 5), 1)
    else:
        return img


# 抖动
def jitter(img):
    """
    jitter
    """
    w, h, _ = img.shape
    if h > 10 and w > 10:
        thres = min(w, h)
        s = int(random.random() * thres * 0.01)
        src_img = img.copy()
        for i in range(s):
            img[i:, i:, :] = src_img[:w - i, :h - i, :]
        return img
    else:
        return img


# 高斯噪声
def add_gasuss_noise(image, mean=0, var=0.1):
    """
    Gasuss noise
    """

    noise = np.random.normal(mean, var**0.5, image.shape)
    out = image + 0.5 * noise
    out = np.clip(out, 0, 255)
    out = np.uint8(out)
    return out


# 随机裁剪
def get_crop(image):
    """
    random crop
    """
    h, w, _ = image.shape
    top_min = 1
    top_max = 8
    top_crop = int(random.randint(top_min, top_max))
    top_crop = min(top_crop, h - 1)
    crop_img = image.copy()
    ratio = random.randint(0, 1)
    if ratio:
        crop_img = crop_img[top_crop:h, :, :]
    else:
        crop_img = crop_img[0:h - top_crop, :, :]
    return crop_img


# 扭曲
def random_distortion(grid_width, grid_height, magnitude,images):
    """
        扭曲
    """

    h, w, _ = images.shape
    horizontal_tiles = grid_width
    vertical_tiles = grid_height
    width_of_square = int(floor(w / float(horizontal_tiles)))
    height_of_square = int(floor(h / float(vertical_tiles)))
    width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
    height_of_last_square = h - (height_of_square * (vertical_tiles - 1))

    dimensions = []

    for vertical_tile in range(vertical_tiles):
        for horizontal_tile in range(horizontal_tiles):
            if vertical_tile == (vertical_tiles - 1) and horizontal_tile == (horizontal_tiles - 1):
                dimensions.append([horizontal_tile * width_of_square,
                                   vertical_tile * height_of_square,
                                   width_of_last_square + (horizontal_tile * width_of_square),
                                   height_of_last_square + (height_of_square * vertical_tile)])
            elif vertical_tile == (vertical_tiles - 1):
                dimensions.append([horizontal_tile * width_of_square,
                                   vertical_tile * height_of_square,
                                   width_of_square + (horizontal_tile * width_of_square),
                                   height_of_last_square + (height_of_square * vertical_tile)])
            elif horizontal_tile == (horizontal_tiles - 1):
                dimensions.append([horizontal_tile * width_of_square,
                                   vertical_tile * height_of_square,
                                   width_of_last_square + (horizontal_tile * width_of_square),
                                   height_of_square + (height_of_square * vertical_tile)])
            else:
                dimensions.append([horizontal_tile * width_of_square,
                                   vertical_tile * height_of_square,
                                   width_of_square + (horizontal_tile * width_of_square),
                                   height_of_square + (height_of_square * vertical_tile)])

    last_column = []
    for i in range(vertical_tiles):
        last_column.append((horizontal_tiles-1)+horizontal_tiles*i)

    last_row = range((horizontal_tiles * vertical_tiles) - horizontal_tiles, horizontal_tiles * vertical_tiles)

    polygons = []
    for x1, y1, x2, y2 in dimensions:
        polygons.append([x1, y1, x1, y2, x2, y2, x2, y1])

    polygon_indices = []
    for i in range((vertical_tiles * horizontal_tiles) - 1):
        if i not in last_row and i not in last_column:
            polygon_indices.append([i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])

    for a, b, c, d in polygon_indices:
        dx = random.randint(-magnitude, magnitude)
        dy = random.randint(-magnitude, magnitude)

        x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]
        polygons[a] = [x1, y1, x2, y2, x3 + dx, y3 + dy, x4, y4]

        x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
        polygons[b] = [x1, y1, x2 + dx, y2 + dy, x3, y3, x4, y4]

        x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
        polygons[c] = [x1, y1, x2, y2, x3, y3, x4 + dx, y4 + dy]

        x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
        polygons[d] = [x1 + dx, y1 + dy, x2, y2, x3, y3, x4, y4]

    generated_mesh = []
    for i in range(len(dimensions)):
        generated_mesh.append([dimensions[i], polygons[i]])

    images = Image.fromarray(cv2.cvtColor(images, cv2.COLOR_BGR2RGB))

    images = images.transform(images.size, Image.MESH, generated_mesh, resample=Image.BICUBIC)
    return cv2.cvtColor(np.asarray(images), cv2.COLOR_RGB2BGR)


# 缩放
def Zoom(min_factor, max_factor, images):

    factor = round(random.uniform(min_factor, max_factor), 2)
    # CV->PIL
    images = Image.fromarray(cv2.cvtColor(images, cv2.COLOR_BGR2RGB))

    def do(image):
        w, h = image.size

        image_zoomed = image.resize((int(round(image.size[0] * factor)),
                                     int(round(image.size[1] * factor))),
                                     resample=Image.BICUBIC)
        w_zoomed, h_zoomed = image_zoomed.size

        return image_zoomed.crop((floor((float(w_zoomed) / 2) - (float(w) / 2)),
                                  floor((float(h_zoomed) / 2) - (float(h) / 2)),
                                  floor((float(w_zoomed) / 2) + (float(w) / 2)),
                                  floor((float(h_zoomed) / 2) + (float(h) / 2))))

    images = do(images)

    return cv2.cvtColor(np.asarray(images), cv2.COLOR_RGB2BGR)


def warp(img):
    """
    warp
    """
    img = np.array(img).copy()
    h, w, _ = img.shape
    new_img = img


    prob_tia = 0          # TIA抖动
    prob_crop = 0         # 随机裁剪
    prob_Zoom = 0.5       # 缩放
    prob_blur = 0         # 模糊
    prob_cvtColor = 0     # 颜色空间转换
    prob_jitter = 0       # 颜色抖动
    prob_gauss_noise = 0  # 高斯噪声
    prob_distort = 0.5    # 扭曲的概率
    # --------------------------- tia扰动 ---------------------------------
    img_height, img_width = img.shape[0:2]
    if random.random() <= prob_tia and img_height >= 20 and img_width >= 20:
        new_img = tia_distort(new_img, random.randint(3, 6))

    if random.random() <= prob_tia and img_height >= 20 and img_width >= 20:
        new_img = tia_stretch(new_img, random.randint(3, 6))

    if random.random() <= prob_tia:
        new_img = tia_perspective(new_img)
    # ---------------------------------------------------------------------
    # 随机裁剪
    img_height, img_width = img.shape[0:2]
    if random.random() <= prob_crop and img_height >= 20 and img_width >= 20:
        new_img = get_crop(new_img)

    # 缩放
    if random.random() <= prob_Zoom:
        new_img = Zoom(min_factor=1.05, max_factor=1.05, images=new_img)

    # 扭曲
    if random.random() <= prob_distort:
        new_img = random_distortion(grid_width=6, grid_height=2, magnitude=3, images=new_img)

    # 模糊
    if random.random() <= prob_blur:
        new_img = blur(new_img)
    # 颜色空间转换
    if random.random() <= prob_cvtColor:
        new_img = cvtColor(new_img)
    # 颜色抖动
    if random.random() <= prob_jitter:
        new_img = jitter(new_img)
    # 高斯噪声
    if random.random() <= prob_gauss_noise:
        new_img = add_gasuss_noise(new_img)

    # if random.random() <= prob:
    #     new_img = 255 - new_img

    return new_img


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


# from https://github.com/pytorch/vision/blob/main/references/classification/transforms.py
class RandomMixup(torch.nn.Module):
    """Randomly apply Mixup to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.
    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for mixup.
            Default value is 1.0. # beta分布超参数
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes, p = 0.5, alpha = 1.0, inplace = False):
        super().__init__()
        assert num_classes > 0, "Please provide a valid positive value for the num_classes."
        assert alpha > 0, "Alpha param can't be zero."

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch, target):
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )
        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        # 建立one-hot标签
        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        # 判断是否进行mixup
        if torch.rand(1).item() >= self.p:
            return batch, target

        # 这里将batch数据平移一个单位，产生mixup的图像对，这意味着每个图像与相邻的下一个图像进行mixup
        # timm实现是通过flip来做的，这意味着第一个图像和最后一个图像进行mixup
        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # 随机生成组合系数
        # Implemented as on mixup paper, page 3.
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        batch_rolled.mul_(1.0 - lambda_param)
        batch.mul_(lambda_param).add_(batch_rolled)  # 得到mixup后的图像

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)  # 得到mixup后的标签

        return batch, target


class RandomCutmix(torch.nn.Module):
    """Randomly apply Cutmix to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    <https://arxiv.org/abs/1905.04899>`_.
    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for cutmix.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        assert num_classes > 0, "Please provide a valid positive value for the num_classes."
        assert alpha > 0, "Alpha param can't be zero."

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch, target):
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W)
            target (Tensor): Integer tensor of size (B, )
        Returns:
            Tensor: Randomly transformed batch.
        """
        if batch.ndim != 4:
            raise ValueError(f"Batch ndim should be 4. Got {batch.ndim}")
        if target.ndim != 1:
            raise ValueError(f"Target ndim should be 1. Got {target.ndim}")
        if not batch.is_floating_point():
            raise TypeError(f"Batch dtype should be a float tensor. Got {batch.dtype}.")
        if target.dtype != torch.int64:
            raise TypeError(f"Target dtype should be torch.int64. Got {target.dtype}")

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).to(dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on cutmix paper, page 12 (with minor corrections on typos).
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        W, H = F.get_image_size(batch)

        # 确定patch的起点
        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        # 确定patch的w和h（其实是一半大小）
        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        # 越界处理
        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        batch[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
        # 由于越界处理， λ可能发生改变，所以要重新计算
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target
