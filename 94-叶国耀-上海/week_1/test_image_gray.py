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

# # 导入图像
# img = plt.imread("lenna.png")
# #
# plt.subplot(221)
# # img = plt.imread("lenna.png")
# # img = cv2.imread("lenna.png", False)
# plt.imshow(img)
# print("---image lenna----")
# print(img)
#
# # 灰度化
# # # 浮点法
# # img_gray =img[:,:,0]*0.3 + img[:,:,1]*0.59 + img[:,:,2]*0.11   #将RGB坐标转化为gray坐标并赋值给新图像
# # #整数法
# # img_gray =(img[:,:,0]*30 + img[:,:,1]*59 + img[:,:,2]*11)/100  #将RGB坐标转化为gray坐标并赋值给新图像
# # #平均法
# img_gray =(img[:,:,0] + img[:,:,1] + img[:,:,2])/3  #将RGB坐标转化为gray坐标并赋值给新图像
# #
# plt.subplot(222)
# plt.imshow(img_gray, cmap='gray')
# print("---image gray----")
# print(img_gray)
#
# # 二值化
# th_2b = 0.5
# img_binary = np.where(img_gray >= th_2b, 1, 0)
# print("-----imge_binary------")
# print(img_binary)
# print(img_binary.shape)
#
# plt.subplot(223)
# plt.imshow(img_binary, cmap='gray')
# plt.show()

# # 导入图像
img = cv2.imread("lenna.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img)
# # 灰度化
# 移位法
img_t = img.astype(int)
img_gray =(img_t[:,:,0]*76 + img_t[:,:,1]*151 + img_t[:,:,2]*28)>>8  #将BGR坐标转化为gray坐标并赋值给新图像
print("---image gray----")
print(img_gray)
img_gray = img_gray.astype(np.uint8)
# # 二值化
img_binary = np.where(img_gray >= 128, 1, 0)
print("---image binary----")
print(img_binary)

plt.subplot(221)
plt.imshow(img)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()