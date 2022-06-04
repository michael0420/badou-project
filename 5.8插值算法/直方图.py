# -*- coding = utf-8 -*-
# @Time : 2022/5/15 20:53
# @Author : L
# @File : 直方图.py
# @Software : PyCharm

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# 灰度图像直方图 plt
img = cv.imread("lenna.png")
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# plt.figure()
# plt.hist(img_gray.ravel(), 256)
# plt.show()


# opencv 方法
# hist = cv.calcHist([img_gray], [0], None, [256], [0, 256])
# plt.figure()  # 新建一个图像
# plt.title("Grayscale Histogram")
# plt.xlabel("Bins")  # X轴标签
# plt.ylabel("# of Pixels")  # Y轴标签
# plt.plot(hist)
# plt.xlim([0, 56])  # 设置x坐标轴范围
# plt.show()


# 彩色直方图
# chans = cv.split(img)
# colors = ("b", "g", "r")
# plt.figure()
# plt.title("Flattened Color Histogram")
# plt.xlabel("Bins")
# plt.ylabel("# of Pixels")
#
# for (chan, color) in zip(chans, colors):
#     hist = cv.calcHist([chan], [0], None, [256], [0, 256])
#     plt.plot(hist, color=color)
#     plt.xlim(0, 256)
# plt.show()


# 灰度图像直方图均衡化
# dst = cv.equalizeHist(img_gray)
# plt.figure()
# plt.hist(dst.ravel(), 256)
# plt.show()
#
# cv.imshow("Histogram Equalization", np.hstack([img_gray, dst]))
# cv.waitKey(0)


# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
(b, g, r) = cv.split(img)
bH = cv.equalizeHist(b)
gH = cv.equalizeHist(g)
rH = cv.equalizeHist(r)
# 合并每一个通道
result = cv.merge((bH, gH, rH))
cv.imshow("dst_rgb", result)

cv.waitKey(0)
cv.destroyAllWindows()
