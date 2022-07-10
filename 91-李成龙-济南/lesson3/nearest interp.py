# coding :utf-8

import cv2
import numpy as np
"""
Task:最邻近插值
Author:91-李成龙-济南
Date:2022-05-13
"""
def function(img, dstH, dstW):
    srcH,srcW,channels = img.shape
    emptyImage = np.zeros((dstH,dstW,channels),np.uint8)
    sh = dstH/srcH  # 确定缩放比例
    sw = dstW/srcW
    for i in range(dstH):
        for j in range(dstW):
            #  int向0取整
            x = int(i/sh)  # 通过目标位置找到对应原（源）图的位置
            y = int(j/sw)  #用取整运算代替对于坐标位置的判断
            emptyImage[i,j]=img[x,y]  #把原（源）图的对应位置的像素值赋给目标对应位置。
    return emptyImage


img = cv2.imread("lenna.png")
dst_img = function(img,800,800)
# print(dst_img)
print(img.shape)
print(dst_img.shape)
cv2.imshow("nearest interp", dst_img)
cv2.imshow("image", img)

cv2.waitKey()
cv2.destroyAllWindows()

