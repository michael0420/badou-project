# coding: gbk

import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
Task:彩色图像的灰度化和二值化
Author:91-李成龙-济南
Date:2022-05-17
"""

'''
equalizeHist—直方图均衡化
函数原型： equalizeHist(src, dst=None)
src：图像矩阵(单通道图像)
dst：默认即可
'''
"""
cv2.calcHist--可以用来统计图像的直方图
image输入图像，传入时应该用中括号[]括起来
channels:：传入图像的通道，如果是灰度图像，那就不用说了，只有一个通道，值为0，如果是彩色图像（有3个通道），
那么值为0,1,2,中选择一个，对应着BGR各个通道。这个值也得用[]传入。
mask：掩膜图像。如果统计整幅图，那么为none。主要是如果要统计部分图的直方图，就得构造相应的炎掩膜来计算。
histSize：灰度级的个数，需要中括号，比如[256]
ranges:像素值的范围，通常[0,256]，有的图像如果不是0-256，比如说你来回各种变换导致像素值负值、很大，则需要调整后才可以。
"""
M = 2
if M == 1:  # 灰度图像均衡化
    # 获取灰度图像
    img = cv2.imread("lenna.png", 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("image_gray", gray)

    # 灰度图像直方图均衡化
    dst = cv2.equalizeHist(gray)

    # 直方图
    hist = cv2.calcHist([dst], [0], None, [256], [0, 256])

    plt.figure()
    plt.hist(dst.ravel(), 256)  # ravel()将数组维度拉成一维
    plt.show()

    cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
    # np.hstack([gray, dst])通过此函数把灰度化的图像和均衡化后的图像拼接成一张图对比显示
    cv2.waitKey()
    cv2.destroyAllWindows()

elif M == 2:
    #  彩色图像直方图均衡化
    img = cv2.imread("lenna.png", 1)
    cv2.imshow("src", img)

    # 彩色图像均衡化,需要分解通道 对每一个通道均衡化
    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    # 合并每一个通道
    result = cv2.merge((bH, gH, rH))
    cv2.imshow("dst_rgb", result)
    cv2.waitKey()
    cv2.destroyAllWindows()
