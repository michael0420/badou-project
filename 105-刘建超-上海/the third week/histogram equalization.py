# -*- coding:gbk -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
calcHist—计算图像直方图
函数原型：calcHist(images, channels, mask, histSize, ranges, hist=None, accumulate=None)
images：图像矩阵，例如：[image]
channels：通道数，例如：0
mask：掩膜，一般为：None
histSize：直方图大小，一般等于灰度级数
ranges：横轴范围
'''

'''
equalizeHist—直方图均衡化
函数原型： equalizeHist(src, dst=None)
src：图像矩阵(单通道图像)
dst：默认即可
'''

# 灰度图像直方图均衡化
img = cv2.imread("lenna.png")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hist_gray = cv2.equalizeHist(gray_img)
# plt.figure()
plt.subplot(2, 1, 1)
plt.hist(gray_img.ravel(), 256)
plt.subplot(2, 1, 2)
plt.hist(hist_gray.ravel(), 256)
plt.show()
cv2.imshow("Histogram Equalization", np.hstack([gray_img, hist_gray]))
cv2.waitKey(0)

# 彩色图像直方图均衡化
img = cv2.imread("lenna.png", 1)
(b, g, r) = cv2.split(img)
hist_b = cv2.equalizeHist(b)
hist_g = cv2.equalizeHist(g)
hist_r = cv2.equalizeHist(r)
result_img = cv2.merge((hist_b, hist_g, hist_r))
cv2.imshow("Histogram Equalization", np.hstack([img, result_img]))
cv2.waitKey()

# # 灰度图像直方图，方法一
# img = cv2.imread("lenna.png", 1)
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # plt.figure()
# plt.subplot(2,2,1)
# plt.hist(gray_img.ravel(),256)
#
# # 灰度图像直方图，方法二
# height, width, channel = img.shape
# src_hist = cv2.calcHist([gray_img], [0], None, [256], [0,256])
# # plt.figure()
# plt.subplot(2,2,2)
# plt.title("Gray Histogram")
# plt.xlabel("Bins")
# plt.ylabel("# of Pixels")
# plt.plot(src_hist)
# plt.xlim([0,256])
#
# # 彩色图像直方图
# chans = cv2.split(img)
# colors = ["b", "g", "r"]
# # plt.figure()
# plt.subplot(2, 2, 3)
# plt.title("Color Histogram")
# plt.xlabel("Bins")
# plt.ylabel("# of Pixels")
#
# for (chan, color) in zip(chans, colors):
#     hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
#     plt.plot(hist, color=color)
#     plt.xlim([0, 256])
# plt.show()
