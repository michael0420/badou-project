import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from PIL import Image

# 灰度图像直方图一
img = cv2.imread("lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure()
plt.hist(gray.ravel(), 256)
plt.show()

# 灰度图像直方图二
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
plt.figure()
plt.plot(hist)
# 设置x坐标轴范围
plt.xlim([0, 256])
plt.show()

# 彩色图像直方图
image = cv2.imread("lenna.png")
channels = cv2.split(image)
colors = ("blue", "green", "red")
plt.figure()
for (chan, color) in zip(channels, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])
plt.show()
