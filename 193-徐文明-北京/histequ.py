import cv2
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter

'''
calcHist—计算图像直方图
函数原型：calcHist(images, channels, mask, histSize, ranges, hist=None, accumulate=None)
images：图像矩阵，例如：[image]
channels：通道数，例如：0
mask：掩膜，一般为：None
histSize：直方图大小，一般等于灰度级数
ranges：横轴范围
'''


# 灰度图像直方图
# 获取灰度图像
# gray = cv2.imread("lenna.jpeg", 1)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2Luv)#COLOR_BGR2GRAY
#cv2.imshow("image_gray", gray)

# 灰度图像的直方图，方法一

# gray = cv2.imread("lenna.jpeg", 1)
# plt.figure()
# plt.hist(gray.ravel(), 256)
# plt.show()


# 灰度图像的直方图, 方法二,线形图
'''
gray = cv2.imread("lenna.jpeg", 1)
hist = cv2.calcHist([gray],[0],None,[256],[0,256])
plt.figure()#新建一个图像
plt.title("Grayscale Histogram")
plt.xlabel("Bins")#X轴标签
plt.ylabel("# of Pixels")#Y轴标签
plt.plot(hist)
plt.xlim([0,256])#设置x坐标轴范围
plt.show()
'''


#彩色图像直方图
'''
image = cv2.imread("lenna.jpeg")
cv2.imshow("Original",image)
#cv2.waitKey(0)

chans = cv2.split(image)
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
'''


def equhist(img):
    """
    灰度直方图均衡化
    :param img:
    :return:
    api : cv2.equalizeHist(gray)
    """

    dcount = Counter(img.ravel())
    dcountk = sorted(dcount)
    hw_256 = img.ravel().shape[0]/256 #h*w/256

    tgtd = {} # 均衡化后的直方图数据
    num = 0
    for d in dcountk:
        num+=dcount[d]/hw_256
        k = max(round(num)-1,0)
        tgtd[k] = dcount[d]
    return tgtd



img = cv2.imread("lenna.jpeg", 0)
tgtd = equhist(img)
plt.plot([x for x in tgtd.keys()], [x for x in tgtd.values()])

plt.show()




