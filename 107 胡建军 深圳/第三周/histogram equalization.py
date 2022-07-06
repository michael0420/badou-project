import cv2
import numpy as np
import matplotlib.pyplot as plt

# 获取灰度图像
img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#进行灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)
#画直方图
plt.figure(figsize=(18,8))
plt.subplot(121)
plt.title('original hist')
plt.hist(gray.ravel(), 256)

plt.subplot(122)
plt.title('equalization hist')
plt.hist(dst.ravel(), 256)

plt.show()

# cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
# cv2.waitKey(0)


# 彩色图像直方图均衡化
img = cv2.imread("lenna.png")
cv2.imshow("src", img)

# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
(b, g, r) = cv2.split(img)
bh = cv2.equalizeHist(b)
gh = cv2.equalizeHist(g)
rh = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((bh, gh, rh))
cv2.imshow("dst_rgb", result)
cv2.waitKey(0)
