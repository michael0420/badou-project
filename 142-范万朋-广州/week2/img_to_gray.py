# -*- codeing = utf-8 -*-
# @Time: 2022/4/22 12:34
# @Author: 棒棒朋
# @File: img_to_gray.py
# @Software: PyCharm
"""
实现图像二值化、灰度化
"""
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
import cv2


# 灰度化
def img_to_gray(img_path):
    """
    图片灰度化
    :param img_path: 图片路径
    """
    img = cv2.imread(img_path)  # 读取图片

    img_gray = rgb2gray(img)
    plt.subplot(121)
    plt.title("gray")
    plt.imshow(img_gray, cmap='gray')
    return img_gray


def img_to_binary(gray_img):
    """
    图像二值化
    :param gray_img: 灰度图后的图片
    """
    img_binary = np.where(gray_img >= 0.5, 1, 0)
    plt.subplot(122)
    plt.imshow(img_binary, cmap='gray')
    plt.title("binary")
    plt.show()


if __name__ == '__main__':
    img_path = "lenna.png"
    gray_img = img_to_gray(img_path=img_path)
    img_to_binary(gray_img)
