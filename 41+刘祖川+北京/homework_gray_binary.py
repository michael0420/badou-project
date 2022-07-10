# -*- coding: utf-8 -*-
import numpy as np
import cv2

#灰度化
img = cv2.imread("lenna.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("img_gray", img_gray)
cv2.waitKey(3000)

#二值化
#img_binary = np.where(img_gray >= 127.0, 255.0, 0)
ret,thresh = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
cv2.imshow("img_binary", thresh)
cv2.waitKey(3000)

