
'''
equalizehist  直方图均衡化
函数原型：e（scr，dst = None）
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
'''
# 获取灰度图像
img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("image_gray", gray)
#灰度直方图均衡化
dst = cv2.equalizeHist(gray)

#绘制 直方图
hist = cv2.calcHist([dst],[0],None,[256],[0,256])

plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)
'''

#彩色直方图均衡化
img = cv2.imread("lenna.png")
cv2.imshow("scr", img)

#分解通道
(b,g,r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
#合并通道
result = cv2.merge((bH, gH, rH))
cv2.imshow("dst_rgb", result)

cv2.waitKey(0)
chans = cv2.split(result)    #cv2.split函数分离出的B、G、R是单通道图像
colors = ("b","g","r")
plt.figure()
plt.title("123")
plt.xlabel("bins")
plt.xlabel("of px")

for (chan,color) in zip(chans,colors):
  hist = cv2.calcHist([chan], [0], None, [256], [0,256])
  plt.plot(hist,color = color)   #每个通道绘制一条线
  plt.xlim([0,256])
plt.show()
