from captcha.image import ImageCaptcha
import random as rd
import os

# 该文件用于随机生成大量的验证码


nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
lower_char = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
upper_char = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


# def get_width():
#     return int(100 + 40 * rd.random())
#
#
# def get_height():
#     return int(45 + 20 * rd.random())


# 获取验证码标签
def get_string():
    string = ""
    for i in range(4):
        select = rd.randint(1, 3)
        if select == 1:
            index = rd.randint(0, 9)
            string += nums[index]
        elif select == 2:
            index = rd.randint(0, 25)
            string += lower_char[index]
        else:
            index = rd.randint(0, 25)
            string += upper_char[index]
    return string


def get_captcha(num, path):
    if not os.path.exists(path):
        os.makedirs(path)
    font_sizes = [x for x in range(40, 45)]
    for i in range(num):
        print(i)
        imc = ImageCaptcha(100, 40,fonts=None, font_sizes=font_sizes)
        name = get_string()
        image = imc.generate_image(name)
        image.save(os.path.join(path,name + ".png"))


if __name__ == '__main__':
    get_captcha(10000, "./data/train")
    get_captcha(2000, "./data/test")
