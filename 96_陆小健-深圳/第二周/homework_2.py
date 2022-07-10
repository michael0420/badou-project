# -*- coding: utf-8 -*-
"""
@author: Michael

彩色图像的灰度化、二值化
"""
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 1_灰度化
img_0 = cv2.imread("lenna.png")
img = img_0[:, :, [2, 1, 0]].copy()
fig_1 = plt.figure(num=1,figsize=(10, 10))  #facecolor='black'
ax1 = fig_1.add_subplot(2,3,1)
ax1.title.set_text('Original')
ax1.imshow(img)

h,w = img.shape[:2]                               #获取图片的high和wide
img_gray = np.zeros([h,w],img.dtype)                   #创建一张和当前图片大小一样的单通道图片
for i in range(h):
    for j in range(w):
        m = img[i,j]                             #取出当前high和wide中的BGR坐标,每一个像素点
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)   #将BGR坐标转化为gray坐标并赋值给新图像
# cv2.imshow("image show gray",img_gray)
ax2 = fig_1.add_subplot(2,3,2)
ax2.title.set_text('gray_image')
ax2.imshow(img_gray, cmap='gray')

gray_image = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)  # 用OpenCV的灰度化函数
ax3 = fig_1.add_subplot(2,3,3)
ax3.title.set_text('opencv_gray_image')
ax3.imshow(gray_image, cmap='gray')


# 2_二值化
rows, cols = img_gray.shape[:2] 
print(img_gray[10, 10])
for i in range(rows):
    for j in range(cols):
        if (img_gray[i, j] <= 120):
            img_gray[i, j] = 0
        else:
            img_gray[i, j] = 1
# img_binary = np.where(img_gray >= 120, 1, 0)
ax5 = fig_1.add_subplot(2,3,5)
ax5.title.set_text('img_binary')
ax5.imshow(img_gray, cmap='gray')

(thresh,opencv_binary) = cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY) # 用OpenCV的二值化函数
ax6 = fig_1.add_subplot(2,3,6)
ax6.title.set_text('opencv_binary')
ax6.imshow(opencv_binary, cmap='gray')
# plt.savefig('image/coin_result.jpg')
plt.show()
