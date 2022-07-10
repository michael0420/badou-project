import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./lenna.png')
print(img.shape)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(18,8))
plt.subplot(221)
plt.hist(gray.ravel(), 256)


#彩色图像直方图
'''
calcHist—计算图像直方图
函数原型：calcHist(images, channels, mask, histSize, ranges, hist=None, accumulate=None)
images：图像矩阵，例如：[image]
channels：通道数，例如：0
mask：掩膜，一般为：None
histSize：直方图大小，一般等于灰度级数
ranges：横轴范围
'''
chans = cv2.split(img)
colors = ('b', 'g', 'r')
plt.subplot(222)
plt.title("Flattened Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

for (chan, color) in zip(chans, colors):
    hist = cv2.calcHist([chan],[0],None,[256],[0,256])
    print(hist)
    plt.plot(hist, color = color)
    plt.xlim([0,256])
plt.show()


