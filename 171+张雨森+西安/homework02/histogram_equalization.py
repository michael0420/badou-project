# -*- coding:utf-8 -*-
"""
作者：YSen
日期：2022年05月18日
功能：灰度图像直方图均衡化 v1.0
    (原始的直方图均衡化)
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def cv_show(name, img):
    """
    图像的显示，创建一个窗口
    利用opencv库的一些指令显示图像，集成成为一个函数
    :param name: 创建的窗口的名字,是一个字符串类型；
    :param img:传入的需要显示的图像名字，是一个变量名
    :return:None
    """
    cv2.imshow(name, img)
    cv2.waitKey(0)  # 等待时间，毫秒级的等待时间，0表示当按下任意键时终止窗口显示
    cv2.destroyAllWindows()  # 进行触发关闭窗口

'''
calcHist-计算图像直方图
函数原型：calcHist(images,channels,mask,histSize,ranges,hist=None,accumulate=None)
images：图像矩阵，例如：[image]
channels：通道数，例如：0
mask：掩膜，一般为：None
histSize：直方图大小，一般等于灰度级数
ranges：横轴范围
'''

# 获取灰度图像
img = cv2.imread("../data/cat.jpg", 0)

"""
最开始的结果显示--1

# 灰度图像的直方图
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.figure()  # 新建一个图像
plt.title("Grayscale Histogram")
plt.xlabel("Bins")  # X轴标签
plt.ylabel("# of Pixels")  # Y轴标签
plt.plot(hist)
plt.xlim([0, 256])  # 设置x坐标轴范围
plt.show()
"""


'''
equalizeHist—直方图均衡化
函数原型： equalizeHist(src, dst=None)
src：图像矩阵(单通道图像)
dst：默认即可
'''

# 灰度图像直方图均衡化
dst = cv2.equalizeHist(img)

# 计算原图和均衡化后的直方图
cul_img = cv2.calcHist([img], [0], None, [256], [0, 256])
cul_dst = cv2.calcHist([dst], [0], None, [256], [0, 256])

"""
最开始的结果显示--2

plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()
"""

# 直方图均衡化的结果展示
fig, axex = plt.subplots(nrows=2, ncols=2, figsize=[10, 8], dpi=100)
axex[0][0].imshow(img, cmap=plt.cm.gray)
axex[0][0].set_title("原图")
axex[0][1].imshow(dst, cmap=plt.cm.gray)
axex[0][1].set_title("均衡化的结果")
axex[1][0].plot(cul_img)
axex[1][0].grid()
axex[1][1].plot(cul_dst)
axex[1][1].grid()
plt.show()

# cv_show("Histogram Equalization", np.hstack([img, dst]))
# cv2.imshow("Histogram Equalization", np.hstack([img, dst]))
# cv2.waitKey(0)
