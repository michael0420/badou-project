# encoding=gbk

import cv2
import matplotlib.pyplot as plt


# 读取原始图像
image_original = cv2.imread("lenna.png")
# 原始图像RGB转化为Gray
image_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
# 显示原始图像灰度直方图
plt.subplot(221)
plt.hist(image_gray.ravel(), 256)

# 原始图像灰度直方图均衡化
hist_gray = cv2.equalizeHist(image_gray)
# 显示原始图像灰度均衡化后的直方图
plt.subplot(222)
plt.hist(hist_gray.ravel(), 256)
plt.show()

# 显示原始图像
cv2.imshow("image_original", image_original)

# 彩色图像直方图均衡化，需要先分解通道，再对每一个通道做均衡化
(b, g, r) = cv2.split(image_original)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((bH, gH, rH))

cv2.imshow("rgb_hist_equalize", result)
cv2.waitKey(0)

