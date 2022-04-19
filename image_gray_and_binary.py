# ! /usr/bin/python
# -*- coding: utf8 -*-
# codeBy:daoru.zhu

from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2

image_path='lenna.png'                      # 图片文件路径

def image_gray_from_cv():
    img = cv2.imread("lenna.png")           # cv读取图片通道顺序为BGR
    h, w = img.shape[:2]                    # 获取图片的高度和宽度
    print(h,w)
    img_gray = np.zeros([h, w], img.dtype)  # 创建一张和读取图片大小一样的单通道图片（像素点值为0）
    for i in range(h):
        for j in range(w):
            channel_arr = img[i, j]                   # 取出当前高度和宽度中的BGR三通道的像素值
            img_gray[i, j] = int(channel_arr[0] * 0.11 + channel_arr[1] * 0.59 + channel_arr[2] * 0.3)  # 将BGR坐标转化为gray坐标并赋值给新图像
    print(img_gray)
# 图像灰度化
def image_gray():
    plt.subplot(222)
    plt.imshow(image_gray_info, cmap='gray')
# 图像二值化
def image_binary():
    image_b=np.where(image_gray_info>0.5,1,0)
    plt.subplot(223)
    plt.imshow(image_b, cmap='gray')
if __name__ == '__main__':
    plt.subplot(221)
    img = plt.imread("lenna.png")
    plt.imshow(img)
    img_info = plt.imread(image_path)  # 读取图片文件并获取文件相关信息
    image_gray_info = rgb2gray(img_info)  # 图片文件的灰度化信息
    print(img_info)
    print(image_gray_info)
    image_gray()
    image_binary()
    plt.show()