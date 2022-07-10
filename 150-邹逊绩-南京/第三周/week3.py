# coding=utf-8
# 1.最邻近插值 2.双线性插值 3.中心重叠重叠  4. 直方图均衡化


import cv2
import numpy as np
import matplotlib.pyplot as plt

ww = 0
hh = 0
cc = 0


# 最邻近插值
def nearest(img, w, h, c):
    newimg = np.zeros((w, h, c), np.uint8)
    for h2 in range(h):
        for w2 in range(w):
            neww = int(w2 / w * ww)
            newh = int(h2 / h * hh)

            newimg[h2, w2] = img[newh, neww]
    cv2.imwrite("nearest.jpg", newimg)

# 双线性插值
def line2(img, w, h, c):
    newimg = np.zeros((w, h, c), np.uint8)
    for h2 in range(h):
        for w2 in range(w):
            w1 = w2
            if w1 == 0:
                w1 = 1
            h1 = h2
            if h1 == 0:
                h1 = 1

            ltw = int(w1 / w * ww + 0.5)
            lth = int(h1 / h * hh + 0.5)
            lbw = int(w1 / w * ww + 0.5)
            lbh = int(h2 / h * hh + 0.5)
            rtw = int(w2 / w * ww + 0.5)
            rth = int(h1 / h * hh + 0.5)
            rbw = int(w2 / w * ww + 0.5)
            rbh = int(h2 / h * hh + 0.5)

            for c1 in range(c):
                newimg[h2, w2, c1] = int((int(img[lth, ltw, c1]) + int(img[lbh, lbw, c1]) + int(img[rth, rtw, c1]) + int(img[rbh, rbw, c1])) / 4)
    cv2.imwrite("line2.jpg", newimg)

# 中心重叠
def centerr(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    maxh = h1
    if maxh < h2:
        maxh = h2
    maxw = w1
    if maxw < w2:
        maxw = w2

    newimg = np.zeros((maxh, maxw, cc), np.uint8)

    if h1 * w1 < h2 * w2:
        drawImg(img2, newimg)
        drawImg(img1, newimg)
    else:
        drawImg(img1, newimg)
        drawImg(img2, newimg)
    cv2.imwrite("centerr.jpg", newimg)

def drawImg(src, dst):
    h1, w1 = src.shape[:2]
    h2, w2 = dst.shape[:2]

    # h2 >= h1  w2 >= w1
    offh = int((h2 - h1) / 2)
    offw = int((w2 - w1) / 2)

    for w in range(w1):
        for h in range(h1):
            dst[h + offh, w + offw] = src[h, w]

# 直方图均衡化
def histogram(img):
    # # 实际分布字典表
    # dict_curr = {}
    #
    # # 标准分布字典表
    # dict_std = {}
    '''
    考虑
        0-30 225-255 各占比：  10
        95-161  50
        70-95， 161-225： 15
    '''
    #
    # h, w = img.shape[:2]
    # total = h * w
    #
    # # 获取实际分布
    # for h1 in range(h):
    #     for w1 in range(w):
    #         dict_curr[img[h, w]] += 1
    # for i in range(256):
    #     dict_curr[i] = dict_curr[i] / total


    rk = img.flatten()

    # 原始图像灰度直方图
    plt.hist(rk, 256, [0, 255], color='r')
    cv2.imshow("原图像", img)

    # 直方图均衡化
    imgDst = T(rk)[img]
    cv2.imwrite("hist.jpg", imgDst)
    plt.hist(imgDst.flatten(), 256, [0, 255], color='b')

    plt.show()

# 计算累计分布函数
def C(rk):
    # 读取图片灰度直方图
    # bins为直方图直方柱的取值向量
    # hist为bins各取值区间上的频数取值
    hist, bins = np.histogram(rk, 256, [0, 256])
    # 计算累计分布函数
    return hist.cumsum()

# 计算灰度均衡化映射
def T(rk):
    cdf = C(rk)
    # 均衡化
    cdf = (cdf - cdf.min()) * (255 - 0) / (cdf.max() - cdf.min()) + 0
    return cdf.astype('uint8')

if __name__ == "__main__":
    # img = cv2.imread("lenna.png")
    # ww, hh, cc = img.shape
    # w1 = 800
    # h1 = 800
    # nearest(img, w1, h1, cc)
    # line2(img, w1, h1, cc)
    #
    # img1 = cv2.imread("tiger.jpg")
    # img2 = cv2.imread("github.jpg")
    # centerr(img1, img2)

    img = cv2.imread("lenna.png", cv2.IMREAD_REDUCED_GRAYSCALE_8)
    histogram(img)

