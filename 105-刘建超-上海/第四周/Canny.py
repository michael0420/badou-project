#!/usr/bin/python
# encoding=utf-8

import cv2

'''
cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient ]]])   
必要参数：
第一个参数是需要处理的原图像，该图像必须为单通道的灰度图；
第二个参数是滞后阈值1；
第三个参数是滞后阈值2。
'''
img = cv2.imread("lenna.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_canny = cv2.Canny(img_gray, 150, 200)
cv2.imshow("Canny", img_canny)
cv2.waitKey()
cv2.destroyAllWindows()
