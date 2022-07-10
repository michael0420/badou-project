#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# !@DATE :2022/04/21 11:36
# !@AUTHOR lzh
# !@FILE :.py

import cv2
import numpy as np
import matplotlib.pyplot as plt

img_bgr = cv2.imread("lenna.png")
h,w=img_bgr.shape[0:2]

# 原始图像
img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
plt.subplot(2,2,1)
plt.imshow(img_rgb)
plt.title("rgb")

# 灰度化
# gray = r*0.3 + g*0.59 + b*0.11
gray = np.zeros((h,w,1),np.uint8)
for i in range(h):
    for j in range(w):
        (b,g,r) = img_bgr[i,j]
        b,g,r = int(b),int(g),int(r)
        num = r*0.3 + g*0.59 + b*0.11
        gray[i,j] = int(num)
plt.subplot(2,2,2)
#cv2.imshow("gray",gray)
plt.imshow(gray,cmap='gray')
plt.title("gray")

cv2_gray = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)
plt.subplot(2,2,3)
plt.imshow(cv2_gray,cmap='gray')
plt.title("cv2_gray")

# 二值化

binary = np.zeros((h,w,1),np.uint8)
for i in range(h):
    for j in range(w):
        num = cv2_gray[i,j]
        binary[i,j] = 255 if num>125 else 0
#cv2.imshow("binary",binary)
# binary=np.where(cv2_gray>=125,1,0)
plt.subplot(2,2,4)
plt.imshow(binary,cmap='gray')
plt.title("binary")

plt.show()
