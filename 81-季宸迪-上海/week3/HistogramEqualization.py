# 直方图， Histgram

import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread("lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

'''
calcHist—计算图像直方图
函数原型：calcHist(images, channels, mask, histSize, ranges, hist=None, accumulate=None)
images：图像矩阵，例如：[image]
channels：通道数，例如：[0]
mask：掩膜，一般为：None
histSize：直方图大小，一般等于灰度级数，例如： [256]
ranges：横轴范围 例如：[0,256]，左闭右开
'''

# 灰度图的直方图和均衡化
# opencv方法
hist = cv2.calcHist([gray], [0], None, [256], [0,256])
plt.figure()
plt.title("Gray Hist by cv2.calcHist")
plt.xlabel("gray")
plt.ylabel("Number of Pixels")
plt.plot(hist)
plt.xlim([0,256])
plt.show()
# plt方法
plt.figure()
plt.hist(gray.ravel(),256)
plt.show()

# 直方图均衡化，图像增强
gray_eq = cv2.equalizeHist(gray)
hist_eq = cv2.calcHist([gray_eq],[0],None,[256],[0,256])
plt.figure()
plt.title("Gray Equalized Hist by cv2.calcHist")
plt.xlabel("gray equalized")
plt.ylabel("Number of Pixels")
plt.plot(hist_eq)
plt.xlim([0,256])
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([gray, gray_eq]))
cv2.waitKey(0)

channels = cv2.split(img)
colors = ('b','g','r')
plt.figure()
plt.title("color Hist")
plt.xlabel("Color")
plt.ylabel("Number of Pixels")
for channel, color_c in zip(channels, colors):
    hist_color = cv2.calcHist([channel],[0],None,[256],[0,256])
    plt.plot(hist_color, color = color_c)
    plt.xlim([0,256])
plt.show()


b_eq = cv2.equalizeHist(channels[0])
g_eq = cv2.equalizeHist(channels[1])
r_eq = cv2.equalizeHist(channels[2])
result = cv2.merge((b_eq, g_eq, r_eq))
cv2.imshow("color equalized", result)
cv2.waitKey(0)
