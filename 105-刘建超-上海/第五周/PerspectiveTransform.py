#!/usr/bin/python
# coding=utf-8

import numpy as np
import cv2

img = cv2.imread("photo1.jpg")
img_copy = img.copy()
print("img_shape:",img.shape)
# 注意这里src和dst输入的坐标顺序是(列，行)
src = np.float32([[207, 152], [517, 285], [16, 602], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

m = cv2.getPerspectiveTransform(src, dst)  # 获取透视变换矩阵
print("WarpMatrix:\n", m)
img_result = cv2.warpPerspective(img_copy, m, (337, 488))
cv2.imshow("src", img)
cv2.imshow("result", img_result)
cv2.waitKey()