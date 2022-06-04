# -*- coding = utf-8 -*-
# @Time : 2022/5/3 12:57
# @Author : L
# @File : 最邻近插值.py
# @Software : PyCharm

import cv2 as cv
import numpy as np


# 最邻近插值  以放大比例
def nni(img, fx=1, fy=1):
    # 获取图像的高、宽、通道数
    H, W, C = img.shape

    # 已知放大比例，计算放大后的图像大小
    aH = int(fx * H)
    aW = int(fy * W)

    y = np.arange(aH).repeat(aW).reshape(aW, -1)

    x = np.tile(np.arange(aW), (aH, 1))

    y = np.round(y / fy).astype(np.int64)
    x = np.round(x / fx).astype(np.int64)

    out = img[y, x]

    return out

# 已知放大后分辨率
def nnii(img, aH=1000, aW=1000):
    H, W, C = img.shape

    # 创建放大后的数组
    emptyImage = np.zeros((aH, aW, C), np.uint8)
    # 计算放大比例
    fx = aH / H
    fy = aW / W

    for i in range(aH):
        for j in range(aW):
            x = int(i / fx)
            y = int(j / fy)
            emptyImage[i, j] = img[x, y]
    return emptyImage


if __name__ == '__main__':
    img = cv.imread("lenna.png")

    # out = nni(img)
    # cv.imshow("Image", out)

    emptyImage = nnii(img, aH=700, aW=700)
    cv.imshow("Image1", emptyImage)

    cv.waitKey(0)
    cv.destroyAllWindows()
