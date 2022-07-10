# -*- coding:utf-8 -*-
"""
作者：YSen
日期：2022年05月04日
功能：第一次作业
        1.实现灰度化
        2.实现图像的二值化
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

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


def RGB_adjust(img):
    """
    由于cv2与matplotlib的显示模式不一致,opencv读取的彩色图像是BGR格式，Matplotlib显示彩色图像是RGB格式；
    因此需要调整BGB显示模式使得利用matplotlib的plt.imshow显示cv2图像时颜色不会错乱
    :param img:输入用opencv读入的图像
    :return:返回新的RGB模式下的图像（该图像仅用于matplotlib的plt.imshow进行显示）
    """
    b, g, r = cv2.split(img)
    img2 = cv2.merge([r, g, b])

    return img2


img1 = cv2.imread('../Data/lenna.png')
cv_show('imgOriginal', img1)

# 读入灰度图像
"""
读入图像的属性，彩色或者灰度图
cv2.IMREAD_COLOR:彩色图像
cv2.IMREAD_GRAYSCALE:灰度图像
"""
img2 = cv2.imread('../Data/lenna.png', cv2.IMREAD_GRAYSCALE)
cv_show("gray_img", img2)

# =====================实现图像的灰度化=========================
# 方法一：直接调用opencv库的接口
img3 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
cv_show("BGR2gary", img3)

# 方法二：运算的方法
img4 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
# print("img4.shape:{}".format(img4.shape))
R = img4[:, :, 0]
G = img4[:, :, 1]
B = img4[:, :, 2]
# print("R.shape:{}".format(R.shape))
# print("G.shape:{}".format(G.shape))
# print("B.shape:{}".format(B.shape))

gray_img4 = 0.3 * R + 0.59 * G + 0.11 * B
gray_img4 = np.floor(gray_img4)  # 要将计算后的矩阵转换为整数，注意转换整数的方法，不是使用int，而是要符合opencv的规范
gray_img4 = gray_img4.astype(np.uint8)  # array.dtype=np.uint8 或是 array=array.astype(np.uint8)
# print("gray_img4:{}".format(gray_img4))
# print("gray_img4.shape:{}".format(gray_img4.shape))
cv_show("BGR2gary+", gray_img4)

# =====================实现图像的二值化=========================
h, w = img3.shape
img5 = img3 / 255
for i in range(h):
    for j in range(w):
        if img5[i, j] <= 0.5:
            img5[i, j] = 0
        else:
            img5[i, j] = 1

cv_show("binary_img", img5)

# =====================统一打印结果=========================
plt.subplot(231)
img1 = RGB_adjust(img1)
plt.imshow(img1)
plt.title("原图")

plt.subplot(232)
plt.imshow(img2, cmap='gray')
plt.title("直接读入灰度图")

plt.subplot(233)
plt.imshow(img3, cmap='gray')
plt.title("使用接口转换为灰度图")

plt.subplot(234)
plt.imshow(gray_img4, cmap='gray')
plt.title("使用运算转换为灰度图")

plt.subplot(235)
plt.imshow(img5, cmap='gray')
plt.title("转换为二值图")

plt.show()
