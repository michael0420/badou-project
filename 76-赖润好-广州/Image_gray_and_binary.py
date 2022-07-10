# -*- coding:utf8 -*-

"""
彩色图像的灰度化、二值化
"""
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

class Image_Processing():

    def __init__(self):
        self.image = 'lenna.png'

# 灰度化-整数算法
# Gray = (R*30+G*59+B*11)/100
    def RGB_to_Gray_By_Integer(self):
        img = plt.imread(self.image)
        img_gray = np.array(img, dtype=np.float32)   # 将图片转换为 numpy 数组
        img_gray[..., 0] = img_gray[..., 0] * 30.0   # R
        img_gray[..., 1] = img_gray[..., 1] * 59.0   # G
        img_gray[..., 2] = img_gray[..., 2] * 11.0   # B
        img_gray = np.sum(img_gray, axis=2)
        img_gray[..., :] = img_gray[..., :] / 100.0
        return img_gray


# 灰度化-浮点算法
# Gray = R*0.3+G*0.59+B*0.11
    def RGB_to_Gray_By_Float(self):
        img = plt.imread(self.image)
        img_gray = np.array(img)
        img_gray[:, :, 0] = img_gray[:, :, 0] * 0.3   # R
        img_gray[:, :, 1] = img_gray[:, :, 1] * 0.59  # G
        img_gray[:, :, 2] = img_gray[:, :, 2] * 0.11  # B
        img_gray = np.sum(img_gray, axis=2)
        return img_gray


# 灰度化-平均值法
# Gray = (R+G+B)/3
    def RGB_to_Gray_By_Average(self):
        img = plt.imread(self.image)
        img_gray = np.array(img, dtype=np.float32)
        img_gray = np.sum(img_gray, axis=2)
        img_gray[..., :] = img_gray[..., :] / 3.0
        return img_gray


# 二值化
    def RGB_to_Binary(self):
        img = cv2.imread(self.image)
        img_gray = rgb2gray(img)       # 图像灰度化
        img_binary = np.where(img_gray >= 0.5, 1, 0)
        return img_binary


if __name__ == '__main__':
    demo = Image_Processing()

    img_integer = demo.RGB_to_Gray_By_Integer()
    plt.subplot(221)
    plt.title("---image gray by Integer----")
    plt.imshow(img_integer, cmap='gray')
    print(img_integer)

    img_float = demo.RGB_to_Gray_By_Float()
    plt.subplot(222)
    plt.title("---image gray by Float----")
    plt.imshow(img_float, cmap='gray')
    print(img_float)

    img_average = demo.RGB_to_Gray_By_Average()
    plt.subplot(223)
    plt.title("---image gray by Average----")
    plt.imshow(img_average, cmap='gray')
    print(img_average)

    img_binary = demo.RGB_to_Binary()
    plt.subplot(224)
    plt.title("---image binary----")
    plt.imshow(img_binary, cmap='gray')
    print(img_binary)
    plt.show()

