# -*- coding: utf-8 -*-
# 导入库函数
import random

from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as cv

# 加载图像
img=cv.imread("lenna.png") # 默认加载的是 BGR 彩色图像
img_RGB = plt.imread("lenna.png")

# 1. 制图条件预设值
plt.rcParams['font.sans-serif']=['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号

fig,axes=plt.subplots(nrows=3,ncols=3,figsize=(30,30))

for i in range(3):
    for j in range(3):
        axes[i,j].set_xticks([])
        axes[i,j].set_yticks([])
axes[0,0].imshow(img)
axes[0,0].set_title("BGR")

axes[0,1].imshow(img_RGB)
axes[0,1].set_title("RGB")


h,w = img.shape[:2]    #获取图片的high和wide
# 1.灰度化

# 1）cv库函数
img_gray1=cv.imread("lenna.png",0) # 加载为灰度图像
img_gray2=cv.cvtColor(img,cv.COLOR_BGR2GRAY) # 使用图像彩色空间变换函数，将BGR空间转化为灰度图

axes[0,2].imshow(img_gray1,cmap='gray')
axes[0,2].set_title("加载的灰度图")
axes[1,0].imshow(img_gray2,cmap='gray')
axes[1,0].set_title("函数变换")

# 2）分量法：将彩色图像中的三个分量的亮度分别作为三个灰度图像的灰度值。
img_gray3 = np.zeros([h,w],img.dtype)                   #创建一张和当前图片大小一样的单通道图片
for i in range(h):
    for j in range(w):
        m = img[i, j]  # 取出当前high和wide中的BGR坐标
        img_gray3[i,j] = random.choice(m)   #将BGR坐标转化为gray坐标并赋值给新图像

axes[1, 1].imshow(img_gray3,cmap='gray')
axes[1, 1].set_title("分量法")


# 3)最大值法：R=G=B=max(R,G,B),这种方法转换的灰度图像亮度很高。
img_gray4 = np.zeros([h,w],img.dtype)                   #创建一张和当前图片大小一样的单通道图片
for i in range(h):
    for j in range(w):
        m = img[i, j]  # 取出当前high和wide中的BGR坐标
        img_gray4[i,j] = int(max(m[0],m[1],m[2]))   #将BGR坐标转化为gray坐标并赋值给新图像


axes[1,2].imshow(img_gray4,cmap='gray')
axes[1,2].set_title("最大值法")

# 4）平均值法: R=G=B=(R+G+B)/3,这种方法产生的灰度图像比较柔和。
img_gray5 = np.zeros([h,w],img.dtype)                   #创建一张和当前图片大小一样的单通道图片
for i in range(h):
    for j in range(w):
        m = img[i, j]  # 取出当前high和wide中的BGR坐标
        img_gray5[i,j] = int(sum(m)/3)   #将BGR坐标转化为gray坐标并赋值给新图像
axes[2,0].imshow(img_gray5,cmap='gray')
axes[2,0].set_title("平均值法")


# 5）加权平均值法：R=G=B=(w1*R+w2*G+w3*B)/3,由于人眼对绿色最为敏感，红色次之，对蓝色的敏感度最低，
#   故一般情况下，w1=0.299,w2=0.587,w3=0.114
img_gray6 = np.zeros([h,w],img.dtype)                   #创建一张和当前图片大小一样的单通道图片
for i in range(h):
    for j in range(w):
        m = img[i,j]                             #取出当前high和wide中的BGR坐标
        img_gray6[i,j] = int(m[0]*0.114 + m[1]*0.587 + m[2]*0.299)   #将BGR坐标转化为gray坐标并赋值给新图像
axes[2,1].imshow(img_gray6,cmap='gray')
axes[2,1].set_title("加权平均值法")

# 2.二值化
img_binaryzation = np.where(img_gray6 >= 122, 1, 0)
axes[2,2].imshow(img_binaryzation,cmap='gray')
axes[2,2].set_title("二值化")

plt.show()