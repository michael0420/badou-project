# -*- coding = utf-8 -*-
# @Time : 2022/4/18 9:53
# @Author : L
# @File : 转灰度二值化.py
# @Software : PyCharm

import cv2 as cv
import numpy as np


def bgr2gray(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    # 转灰度
    img_gray = b * 0.11 + g * 0.59 + r * 0.3
    out = img_gray.astype(np.uint8)

    return out


def thr(img_gray):
    # 二值化
    img_thr = img_gray / 255
    print(img_thr.shape)
    h, w = img_thr.shape[:2]
    for i in range(h):
        for j in range(w):
            if (img_thr[i, j]) < 0.5:
                img_thr[i, j] = 0
            else:
                img_thr[i, j] = 255
    return img_thr


if __name__ == '__main__':
    img = cv.imread("lenna.png")
    out = bgr2gray(img)

    img_thr = thr(out)
    print(img_thr)

    cv.imshow("Image1", img_thr)
    cv.imshow("Image", out)
    cv.waitKey(0)
    cv.destroyAllWindows()
