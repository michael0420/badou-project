# coding :utf-8
"""
Task:彩色图像的灰度化和二值化
Author:91-李成龙-济南
Date:2022-04-21
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from PIL import Image
import cv2

Method = 2  # Method = 1 手推方法  Method = 2 调库方法
if Method == 1:
    image = plt.imread("lena.jpg")
    plt.subplot(221)
    plt.imshow(image)
    print(image)
    # plt.show()
    print("----------方法一：手推----------")
    [h, w] = image.shape[:2]
    # #########实现灰度化###########
    image_gray = np.zeros([h, w], np.uint8)
    for i in range(h):
        for j in range(w):
            img = image[i, j]
            image_gray[i, j]= img[0]*0.3+img[1]*0.59+img[2]*0.11  # 将RGB值转化为灰度值，并赋给新的图像空间。(浮点算法)
            # image_gray[i, j] = (img[0] * 11 + img[1] * 59 + img[2] * 30) / 100 #（整数方法）
            # image_gray[i, j] = (img[0] + img[1] + img[2]) / 3 # 取均值法
            # image_gray[i, j] = img[2]  # 仅取一种颜色
    print("image_gray的灰度值:", image_gray)
    plt.subplot(222)
    plt.imshow(image_gray, cmap="gray")
    # plt.imshow(image_gray)
    # plt.show()

    # ##########实现二值化###########
    # image__gray = rgb2gray(image)
    m, n = image_gray.shape[:2]
    print(image_gray.shape[:2])
    image_binary = np.zeros([m, n], np.uint8)
    for i in range(m):
        for j in range(n):
            if image_gray[i, j] < 127:
                image_gray[i, j] = 0
                image_binary[i, j] = image_gray[i, j]
            else:
                image_gray[i, j] = 1
                image_binary[i, j] = image_gray[i, j]
    # image_binary = np.where(image_gray >= 127, 255, 0)
    print("image_binary二值化的值为", image_binary)
    plt.subplot(223)
    plt.imshow(image_binary, cmap='gray')
    plt.show()
elif Method == 2:
    image = plt.imread("lena.jpg")
    plt.subplot(221)
    plt.imshow(image)
    print("----------方法二：调库----------")
    # ############实现灰度化#############
    image_gray = rgb2gray(image)
    plt.subplot(222)
    plt.imshow(image_gray, cmap="gray")
    # ############实现二值化#############
    image_rgb = cv2.imread("lena.jpg");
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)  # 转成灰度图片
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)    # 二值化
    ret, image_binary = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)
    image__rgb = cv2.cvtColor(image_binary, cv2.COLOR_BGR2RGB)
    plt.subplot(223)
    plt.imshow(image__rgb)
    plt.show()