# -*- coding: utf-8 -*-
import time

import numpy as np
import cv2
import matplotlib


def gray_img(img_file):
    '''
    这是一个将图片灰度化的函数。
    :param img_file: 三通道图片地址
    :return: 单通道数据矩阵
    '''
    img_data = cv2.imread(img_file)     #读取图片
    h, w = img_data.shape[:2]       #获取宽高
    gray_data = np.zeros([h, w], dtype=img_data.dtype)      #准备同宽高的黑色图片
    for i in range(len(img_data)):
        for j in range(len(img_data[i])):
            item_data = img_data[i][j]      #遍历图片每个像素的三通道值列表
            gray_data[i][j] = item_data[0] * 0.3 + item_data[1] * 0.59 + item_data[2] * 0.11        #改三通道列表成单通道值并赋值到新图片上
    return gray_data


def value2(img_file):
    '''
    这是一个将图片二值化的函数
    :param img_file: 三通道图片地址
    :return: 单通道的二值化（0,255）矩阵
    '''
    gray_data = gray_img(img_file)      #获取单通道灰度化矩阵
    h, w = gray_data.shape      #获取矩阵宽高
    val2 = np.zeros([h, w], dtype=gray_data.dtype)      #准备同宽高的黑色图片
    for i in range(h):
        for j in range(w):
            if gray_data[i][j] >= 125:      #遍历矩阵数据，根据值大小对同位置的新矩阵赋值0或者255
                val2[i][j] = 255
            else:
                val2[i][j] = 0
    return val2


gray_data = gray_img('lenna.png')
val2_data = value2('lenna.png')

cv2.imshow('gray',gray_data)
cv2.imshow('val2',val2_data)
cv2.waitKey(8000)