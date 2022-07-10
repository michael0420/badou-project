#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2022/6/13 0:16
@Author  : luoshixiang
@Email   : just_rick@163.com
@File    : 01_image_gray.py
@effect  : 第一次作业: 彩色图像的灰度化、二值化
"""

# from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def bgr2rgb(img):
    # BGR转RGB
    # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 方法1
    # image = img[:, :, (2, 1, 0)]  # 方法2
    image = img[...,::-1]   # 方法3
    return image

def bgr2gray(img):
    # 灰度化
    img = bgr2rgb(img)
    img_gray = rgb2gray(img)
    return img_gray

def rgb2gray(img_rgb):
    # 灰度化
    h, w = img_rgb.shape[:2]  # 获取图片high, wide
    img_gray = np.zeros([h, w], dtype=img_rgb.dtype)  # 创建一张和当前图片大小一样的单通道图片
    # 遍历图片所有像素点
    for i in range(h):
        for j in range(w):
            m = img_rgb[i, j]  # 取出当前high和wide中的BGR坐标
            # img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)  # 将BGR坐标转化为gray坐标并赋值给新图像
            img_gray[i, j] = (m[0] * 11 + m[1] * 59 + m[2] * 30)/100    # 将BGR坐标转化为gray坐标并赋值给新图像
    # print("image show gray: %s" % img_gray)
    # cv2.imshow("image show gray", img_gray)
    # cv2.waitKey(0)
    return img_gray

def rgb2binary(image, threshold):
    """
    二值化
    @param image: rgb图像矩阵
    @param threshold: 阈值
    @return:
    """
    h, w = image.shape[:2]  # 获取图片high, wide
    image_binary = np.zeros((h, w), dtype=image.dtype)
    image_gray = bgr2gray(image)
    for i in range(h):
        for j in range(w):
            if image_gray[i][j] > threshold:
                image_binary[i][j] = 255
            else:
                image_binary[i][j] = 0
    return image_binary

if __name__ == '__main__':

    image_file = 'lenna.png'

    img = cv2.imread(image_file)

    # 原图
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.axis("off")     # 关闭坐标轴
    plt.title("lenna_BGR")
    cv2.imwrite("./lenna_BGR.png", img)

    # BGR转RGB
    image = bgr2rgb(img)
    plt.subplot(2, 2, 2)
    plt.imshow(image)
    plt.axis("off")  # 关闭坐标轴
    plt.title("lenna_RGB")

    # 灰度化
    img_gray = bgr2gray(image)
    plt.subplot(2, 2, 3)
    plt.imshow(img_gray, cmap="gray")   # 这里必须加 cmap='gray', 否则尽管原图像是灰度图，但是显示的是伪彩色图像
    plt.axis("off")     # 关闭坐标轴
    plt.title("lenna_Gray")
    cv2.imwrite("./lenna_Gray.png", img_gray)

    # 二值化
    image_binary = rgb2binary(image, threshold=135) # threshold阀值
    # cv2.imshow("lenna_binary", image_binary)
    # cv2.waitKey(0)
    plt.subplot(2, 2, 4)
    plt.imshow(image_binary, cmap="gray")
    plt.axis("off")  # 关闭坐标轴
    plt.title("lenna_binary")
    plt.show()
    cv2.imwrite("./lenna_binary.png", image_binary)
