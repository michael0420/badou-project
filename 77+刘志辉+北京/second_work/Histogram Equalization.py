#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt



if __name__=="__main__":
    img = cv2.imread("D:\lenna.png", 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    '''
    # 灰度图像直方图均衡化
    dst = cv2.equalizeHist(gray)
    cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
    
    # 直方图
    hist = cv2.calcHist([dst],[0],None,[256],[0,256])
    plt.figure()
    plt.subplot(121)
    plt.plot(hist)
    plt.subplot(122)
    plt.hist(dst.ravel(), 256)
    plt.show()
    '''
    # 彩色图像均衡化,需要分解通道 对每一个通道均衡化
    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    # 合并每一个通道
    result = cv2.merge((bH, gH, rH))
    cv2.imshow("dst_rgb", np.hstack([img, result]))
    plt.figure()
    plt.subplot(131)
    plt.hist(bH.ravel(), 256)
    plt.subplot(132)
    plt.hist(gH.ravel(), 256)
    plt.subplot(133)
    plt.hist(rH.ravel(), 256)
    plt.show()
    cv2.waitKey(0)
