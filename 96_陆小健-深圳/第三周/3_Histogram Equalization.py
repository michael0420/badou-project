import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
equalizeHist直方图均衡化
函数原型 equalizeHist(src, dst=None)
src：图像矩阵（单通道）
dst：默认值
'''
fig_1 = plt.figure(num=1,figsize=(10, 10))  #facecolor='black'
#获取灰度图
img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 灰度图均衡化
dst = cv2.equalizeHist(gray)
cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
#直方图
hist_1 = cv2.calcHist([gray],[0],None,[256],[0,256])
hist_2 = cv2.calcHist([dst],[0],None,[256],[0,256])

#彩色图像直方图
# 彩色图均衡化 分解通道
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并通道
result = cv2.merge((bH, gH, rH))
cv2.imshow("dst_rgb", np.hstack([img, result]))

# 展现灰度图he彩色图原图和均衡化后图像之间的直方图
# fig_1 = plt.figure(num=1,figsize=(10, 10))  #facecolor='black'
ax1 = fig_1.add_subplot(2,2,1)
ax1.title.set_text('gray_calcHist')
ax1.hist(gray.ravel(), 256)
ax2 = fig_1.add_subplot(2,2,2)
ax2.title.set_text('dst_calcHist')
ax2.hist(dst.ravel(), 256)
ax3 = fig_1.add_subplot(2,2,3)
ax3.title.set_text('Flattened Color Histogram_1')
chans_3 = cv2.split(img)
colors = ("b","g","r")
for (chan,color) in zip(chans_3,colors):
    hist_3 = cv2.calcHist([chan],[0],None,[256],[0,256])
    ax3.plot(hist_3,color = color)
ax4 = fig_1.add_subplot(2,2,4)
ax4.title.set_text('Flattened Color Histogram_2')
chans_4 = cv2.split(result)
for (chan,color) in zip(chans_4,colors):
    hist_4 = cv2.calcHist([chan],[0],None,[256],[0,256])
    ax4.plot(hist_4,color = color)
plt.savefig('result.jpg')
plt.show()
cv2.waitKey(0)
