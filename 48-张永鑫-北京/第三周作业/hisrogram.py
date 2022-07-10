import cv2
import numpy as np
from matplotlib import pyplot as plt

# 灰度图像直方图
# 获取灰度图像
img = cv2.imread("img/lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow("image_gray", gray)

# # 灰度图像的直方图
# plt.figure()
# plt.hist(gray.ravel(), 256)
# plt.show()

# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)

# 直方图
hist = cv2.calcHist([dst],[0],None,[256],[0,256])

plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)