# Learner: 王振强
# Learn Time: 2022/3/8 19:02
import cv2
import random
import math
from math import floor, ceil
from PIL import Image, ImageOps, ImageEnhance
import numpy as np


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


# 放大缩小
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



# 平移
def Move(img):
    img_info=img.shape
    height=img_info[0]
    width=img_info[1]

    x = random.randint(-int(width*0.1),int(width*0.1))
    y = random.randint(-int(height*0.2),int(height*0.2))

    mat_translation=np.float32([[1,0,x],[0,1,y]])  #变换矩阵：设置平移变换所需的计算矩阵：2行3列
    #[[1,0,20],[0,1,50]]   表示平移变换：其中x表示水平方向上的平移距离，y表示竖直方向上的平移距离。
    dst=cv2.warpAffine(img,mat_translation,(width,height))  #变换函数
    return dst


if __name__ == '__main__':
    image = 'fig/46.png'
    image = cv2.imread(image)
    # image = Image.open(image)
    print(image.shape)
    # img = random_distortion(grid_width=6, grid_height=2, magnitude=3, images = image)
    #
    # img_blur= Zoom(min_factor=0.9, max_factor=1.1, images = image)
    img = Move(image)


    # print(img)
    cv2.imshow("www",image)
    cv2.imshow("aug img",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()






































