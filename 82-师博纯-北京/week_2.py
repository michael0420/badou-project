# -*- coding: utf-8 -*-
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 灰度化
img = cv2.imread("lenna.png")                   # 读取图片
h, w = img.shape[:2]                            # 获取图片的high和wide
img_gray_float = np.zeros([h, w], img.dtype)    # 创建一张和当前图片大小一样的单通道图片
img_gray_int = np.zeros([h, w], img.dtype)      # 创建一张和当前图片大小一样的单通道图片
img_gray_move = np.zeros([h, w], img.dtype)     # 创建一张和当前图片大小一样的单通道图片
img_gray_average = np.zeros([h, w], img.dtype)  # 创建一张和当前图片大小一样的单通道图片
img_gray_green = np.zeros([h, w], img.dtype)    # 创建一张和当前图片大小一样的单通道图片
for i in range(h):
    for j in range(w):
        m = img[i, j]  # 取出当前high和wide中的BGR坐标
        img_gray_float[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)      # 浮点算法
        img_gray_int[i, j]   = int(m[0] * 11 + m[1] * 59 + m[2] * 30) / 100     # 整数算法
        img_gray_move[i, j]  = int(m[0] * 28 + m[1] * 151 + m[2] * 76) >> 8     # 移位算法
        img_gray_average[i, j] = int( int(m[0]) + int(m[1])  + int(m[2])/ 3)    # 平均算法
        img_gray_green[i, j] = int(m[1])                                        # 仅取绿色
# 灰度图片显示
plt.subplot(4,3,1)
img = plt.imread("lenna.png")
plt.imshow(img)
plt.subplot(4,3,2)
plt.imshow(img_gray_float, cmap='gray')
plt.subplot(4,3,3)
plt.imshow(img_gray_int, cmap='gray')
plt.subplot(4,3,4)
plt.imshow(img_gray_move, cmap='gray')
plt.subplot(4,3,5)
plt.imshow(img_gray_average, cmap='gray')
plt.subplot(4,3,6)
plt.imshow(img_gray_green, cmap='gray')


# 二值化
img_gray = rgb2gray(img)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                    # 使用函数接口得到灰度图二值化
img_binary_float = np.where(img_gray_float >= 125, 255, 0)          # 浮点数算法得到的灰度图进行二值化
img_binary_int = np.where(img_gray_int >= 125, 255, 0)              # 整数算法得到的灰度图进行二值化
img_binary_move = np.where(img_gray_move >= 125, 255, 0)            # 移动算法得到的灰度图进行二值化
img_binary_average = np.where(img_gray_average >= 125, 255, 0)      # 平均算法得到的灰度图进行二值化
img_binary_green = np.where(img_gray_green >= 125, 255, 0)          # 仅取绿色得到的灰度图进行二值化
# 二值化图像显示
plt.subplot(4,3,7)
plt.imshow(img_gray, cmap='gray')
plt.subplot(4,3,8)
plt.imshow(img_binary_float, cmap='gray')
plt.subplot(4,3,9)
plt.imshow(img_binary_int, cmap='gray')
plt.subplot(4,3,10)
plt.imshow(img_binary_move, cmap='gray')
plt.subplot(4,3,11)
plt.imshow(img_binary_average, cmap='gray')
plt.subplot(4,3,12)
plt.imshow(img_binary_green, cmap='gray')
plt.show()
