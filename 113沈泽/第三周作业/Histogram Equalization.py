import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from PIL import Image

# 灰度图直方图均衡化
img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure()
plt.subplot(221)
plt.hist(gray.ravel(), 256)

dst = cv2.equalizeHist(gray)
hist = cv2.calcHist([dst], [0], None, [256], [0, 256])

plt.subplot(222)
plt.hist(dst.ravel(), 256)

plt.subplot(223)
plt.imshow(gray, cmap='gray')

plt.subplot(224)
plt.imshow(dst, cmap='gray')
plt.show()

# 彩色图像直方图均衡化
img = cv2.imread("lenna.png", 1)
plt.figure()
plt.subplot(221)
plt.imshow(img)

(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)

result = cv2.merge((bH, gH, rH))
plt.subplot(222)
plt.imshow(result)

channels = cv2.split(img)
colors = ("blue", "green", "red")
plt.subplot(223)
for (chan, color) in zip(channels, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])

channels1 = cv2.split(result)
colors1 = ("blue", "green", "red")
plt.subplot(224)
for (chan1, color1) in zip(channels1, colors1):
    hist = cv2.calcHist([chan1], [0], None, [256], [0, 256])
    plt.plot(hist, color=color1)
    plt.xlim([0, 256])
plt.show()
