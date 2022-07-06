import cv2
import numpy as np
from matplotlib import pyplot as plt

# 获取灰度图像
img = cv2.imread("../image/lenna.png")
image = plt.imread("../image/lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)

# 灰度图直方图方法一
plt.figure()
plt.subplot(231)
plt.title("Grayscale Histogram 1")
plt.xlabel("Bins")
plt.ylabel("num of Pixels")
plt.hist(gray.ravel(), 256)

# 灰度图直方图方法二
hist = cv2.calcHist([gray],[0],None,[256],[0,256])
plt.subplot(232)
plt.title("Grayscale Histogram 2")
plt.xlabel("Bins")
plt.ylabel("num of Pixels")
plt.plot(hist)
plt.xlim([0,256])

#彩色图像直方图
channels = cv2.split(img)
colors = ("b", "g", "r")
plt.subplot(233)
plt.title("Flattened Color Histogram")
plt.xlabel("Bins")
plt.ylabel("num of Pixels")
for (chan,color) in zip(channels,colors):
    hist = cv2.calcHist([chan],[0],None,[256],[0,256])
    plt.plot(hist,color = color)
    plt.xlim([0,256])


# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)
plt.subplot(234)
plt.hist(dst.ravel(), 256)

# 彩色图像直方图均衡化
plt.subplot(235)
for (chan,color) in zip(channels,colors):
    chan = cv2.equalizeHist(chan)
    hist = cv2.calcHist([chan],[0],None,[256],[0,256])
    plt.plot(hist,color = color)
    plt.xlim([0,256])

# 彩色图像直方图均衡化后图像

plt.subplot(236)
result = cv2.merge((cv2.equalizeHist(channels[0]),cv2.equalizeHist(channels[1]),cv2.equalizeHist(channels[2])))
plt.imshow(result)
plt.show()
cv2.imshow("Pic",np.hstack([img,result]))
cv2.waitKey(0)