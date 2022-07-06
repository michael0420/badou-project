#!/usr/bin/env python
# encoding=gbk

import cv2
import numpy as np
from Day2Code import nearest_interp as near
from matplotlib import pyplot as plt

'''
equalizeHist—直方图均衡化
函数原型： equalizeHist(src, dst=None)
src：图像矩阵(单通道图像)
dst：默认即可
'''

# 获取灰度图像
img = cv2.imread("l_hires.jpg", 1)

# 缩小图像
zoomh = 900
zoomw = 600
zoom = near.function(img,zoomh,zoomw)

cv2.imshow("image_Zoom", zoom)

gray = cv2.cvtColor(zoom, cv2.COLOR_BGR2GRAY)
cv2.imshow("image_gray", gray)

# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)

cv2.imshow("Histogram_luna", dst)

# 直方图
hist = cv2.calcHist([dst],[0],None,[256],[0,256])

plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()




(b, g, r) = cv2.split(zoom)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((bH, gH, rH))

cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))

gray2 = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

cv2.imshow("Histogram Equalization_2", np.hstack([gray2, dst]))
## cv2.imshow("dst_rgb", result)


cv2.waitKey(0)




'''
# 彩色图像直方图均衡化
img = cv2.imread("lenna.png", 1)
cv2.imshow("src", img)

# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((bH, gH, rH))
cv2.imshow("dst_rgb", result)

cv2.waitKey(0)
'''
