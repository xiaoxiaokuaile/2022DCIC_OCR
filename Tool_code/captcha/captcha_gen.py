# -*- coding: UTF-8 -*-
from captcha.image import ImageCaptcha  # pip install captcha
from PIL import Image
import random
import os

"""
    利用 captcha 生成验证码
"""


# 验证码中的字符
# string.digits + string.ascii_uppercase
NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET_1 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
ALPHABET_2 = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
ALL_CHAR_SET = NUMBER + ALPHABET_1 + ALPHABET_2

ALL_CHAR_SET_LEN = len(ALL_CHAR_SET)
# 验证码字符个数
MAX_CAPTCHA = 4


def random_captcha():
    captcha_text = []
    for i in range(MAX_CAPTCHA):
        c = random.choice(ALL_CHAR_SET)
        captcha_text.append(c)
    return ''.join(captcha_text)


# 生成字符对应的验证码
def gen_captcha_text_and_image(width, height, fonts, font_sizes):
    # ImageCaptcha(width=160, height=60, fonts=None, font_sizes=None)
    image = ImageCaptcha(width=width, height=height, fonts=fonts, font_sizes=font_sizes)
    captcha_text = random_captcha()
    captcha_image = Image.open(image.generate(captcha_text))
    return captcha_text, captcha_image


if __name__ == '__main__':
    count = 15000
    img_width = 100 # 生成切片宽度
    img_height = 40 # 生成切片高度
    # 字体 NORMT,Elronmonospace,FUTRFW,HappyMonkey,PAPL,None
    # fonts = None
    fonts = ['./fonts/Elronmonospace.ttf']


    TRAIN_DATASET_PATH = 'dataset' + os.path.sep + 'make_15000'
    path = TRAIN_DATASET_PATH    #通过改变此处目录，以生成 训练、测试和预测用的验证码集
    # 生成目录
    if not os.path.exists(path):
        os.makedirs(path)
    for i in range(count):
        text, image = gen_captcha_text_and_image(img_width,img_height,fonts,[45])
        # filename = str(i) + '_' + text + '.png'
        filename = text + '.png'
        image.save(path  + os.path.sep +  filename)

        if i % 100 == 0:
            print('saved %d : %s' % (i+1,filename))

