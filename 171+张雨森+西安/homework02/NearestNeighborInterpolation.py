# -*- coding:utf-8 -*-
"""
作者：YSen
日期：2022年07月14日
功能：最近邻插值
"""

import cv2
import numpy as np


def cv_show(name, img):
    """
    图像的显示，创建一个窗口
    利用opencv库的一些指令显示图像，集成成为一个函数
    :param name: 创建的窗口的名字,是一个字符串类型；
    :param img:传入的需要显示的图像名字，是一个变量名
    :return:None
    """
    cv2.imshow(name, img)
    cv2.waitKey(0)  # 等待时间，毫秒级的等待时间，0表示当按下任意键时终止窗口显示
    cv2.destroyAllWindows()  # 进行触发关闭窗口


def nearest_neighbor_interpolation(img):
    height, weight, channels = img.shape
    h = height * 3
    w = weight * 3
    emptyImage = np.zeros((h, w, channels), np.uint8)
    sh = h / height  # 放大的比例
    sw = w / weight

    for i in range(h):
        for j in range(w):
            x = int(i/sh)  # 取整
            y = int(j/sw)
            emptyImage[i, j] = img[x, y]  # 用原图像的像素值赋值

    return emptyImage


img = cv2.imread('../Data/cat.jpg')
zoom = nearest_neighbor_interpolation(img)
# print(zoom)
print(zoom.shape)

cv_show("img_origin", img)
cv_show("img", zoom)
