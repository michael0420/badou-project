# cv实现最近领差值、双线性差值、直方图均衡化
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('lenna.png')
cv.imshow('image',img)
# 最近邻差值
res = cv.resize(img,None,fx=0.5,fy=0.5,interpolation=cv.INTER_NEAREST)
cv.imshow('nearest_interp',res)
# 双线性差值
res1 = cv.resize(img,None,fx=0.5,fy=0.5,interpolation=cv.INTER_LINEAR)
cv.imshow('linear_interp',res1)
# 直方图均衡化
img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY) # 首先将图形变为单通道
cv.imshow('image_gray',img_gray)
img_equ = cv.equalizeHist(img_gray)
plt.figure()
plt.subplot(211),plt.hist(img_gray.ravel(),256,[0,256]),plt.title('img_gray.hist')
plt.subplot(212),plt.hist(img_equ.ravel(),256,[0,256]),plt.title('img_equ.hist')
plt.show()

cv.imshow('img_equ',np.hstack((img_gray,img_equ)))
cv.waitKey(0)
