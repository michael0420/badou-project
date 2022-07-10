
'''
calcHist—计算图像直方图
函数原型：calcHist(images, channels, mask, histSize, ranges, hist=None, accumulate=None)
images：图像矩阵，例如：[image]
channels：通道数，例如：0
mask：掩膜，一般为：None
histSize：直方图大小，一般等于灰度级数
ranges：横轴范围
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('lenna.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("gray",gray)
cv2.waitKey(0)
#灰度图直方图方法一
'''
plt.figure()
plt.hist(gray.ravel(),256)
plt.show()
'''
#灰度图直方图方法二
hist = cv2.calcHist([gray],[0],None,[256],[0,256])
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0,256])
plt.show()
#彩色图直方图
chans = cv2.split(img)
colors = ("b","g","r")
plt.figure()
plt.title("Flattened Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

for (chan,color) in zip(chans,colors):
    hist = cv2.calcHist([chan],[0],None,[256],[0,256])
    plt.plot(hist,color = color)
    plt.xlim([0,256])
plt.show()
#灰度图直方图均衡化
dst = cv2.equalizeHist(gray)
cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)
#彩色图直方图均衡化
#分解通道
(b,g,r) = cv2.split(img)
bh = cv2.equalizeHist(b)
gh = cv2.equalizeHist(g)
rh = cv2.equalizeHist(r)
#合并通道
imge = cv2.merge((bh,gh,rh))
cv2.imshow("dst_bgr",imge)
cv2.waitKey(0)