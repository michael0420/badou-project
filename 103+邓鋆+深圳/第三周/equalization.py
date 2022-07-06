import cv2
import numpy as np


# 彩色图像直方图均衡化
img = cv2.imread("w2.png", 1)

# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((bH, gH, rH))

cv2.imshow("dst_rgb", np.hstack([img, result]))
cv2.waitKey(0)


#灰度图
img = cv2.imread("w2.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst = cv2.equalizeHist(gray)
# 直方图
hist = cv2.calcHist([dst],[0],None,[256],[0,256])
cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)