"""
work_2_2:  实现直方图及其均衡化
运用包/模块：PIL、matplotlib.pyplot、numpy、cv2
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

#灰度图像直方图
img = cv2.imread("lenna.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
hist1 = cv2.calcHist([gray],[0],None,[256],[0,256])
plt.figure()#新建一个图像
plt.title("Grayscale Histogram")
plt.xlabel("Bins")#X轴标签
plt.ylabel("Number")#Y轴标签
plt.plot(hist1)
plt.xlim([0,256])#设置x坐标轴范围
plt.show()
# 灰度图像直方图均衡化
plt.figure()
dst = cv2.equalizeHist(gray)
hist_HE = cv2.calcHist([dst],[0],None,[256],[0,256])
plt.title("grayHis_HE")
plt.xlabel("Bins")#X轴标签
plt.ylabel("Number")#Y轴标签
plt.plot(hist_HE)
plt.xlim([0,256])#设置x坐标轴范围
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
# plt.figure()
# plt.hist(gray.ravel(),256)
# plt.show()

#彩色图像直方图
chans = cv2.split(img)
colors = ("b","g","r")
plt.figure()
plt.title("Color image histogram")
plt.xlabel("Bins")
plt.ylabel("Numble")
for (chans,colors) in zip(chans,colors):
    hist2 = cv2.calcHist([chans], [0], None, [256], [0, 256])
    plt.plot(hist2,color = colors)
plt.show()

# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((bH, gH, rH))
cv2.imshow("org dst_HE", np.hstack([img, result]))
cv2.waitKey(0)
