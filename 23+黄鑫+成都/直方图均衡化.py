import numpy as np
import cv2
import matplotlib.pyplot as plt

# 获取灰度图
img = cv2.imread("C:\\Users\\LENOVO\\Desktop\\lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 灰度直方图均衡化
dst = cv2.equalizeHist(gray)

# 直方图
hist = cv2.calcHist([dst], [0], None, [256], [0, 256])  # cv2.calcHist(images,channels,mask,histSize,ranges)
# histSize：直方图大小，一般等于灰度级数
# mask：掩膜，一般为：None
# ranges：横轴范围

plt.figure()       #绘制窗口
plt.hist(dst.ravel(), 256)  # img.ravel()–把多维数组转化成一维数组 横坐标最大值为256
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))  # np.hask()数组水平叠加
cv2.waitKey(0)


#彩色图像直方图均衡化
img=cv2.imread("C:\\Users\\LENOVO\\Desktop\\lenna.png",1)
#img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#cv2.imshow('src',img)

#彩色直方图均衡化，需要分解每一个通道,对每一个通道做均衡化
(b,g,r)=cv2.split(img)
bC=cv2.equalizeHist(b)
gC=cv2.equalizeHist(g)
rC=cv2.equalizeHist(r)

#merge channels
result = cv2.merge((bC,gC,rC))
cv2.imshow('Colorful Histogram Equalization',np.hstack([img,result]))

cv2.waitKey(0)

