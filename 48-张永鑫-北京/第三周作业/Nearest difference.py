import cv2
import numpy as np


def function(img):
    height,width,channels =img.shape  #返回图片的长、宽、通道
    emptyImage=np.zeros((25,25,channels),np.uint8)  #创建一个为零的矩阵数组
    sh=25/height
    sw=25/width
    for channel in range(channels):
        for i in range(25):
            for j in range(25):
                x=int(i/sh)
                y=int(j/sw)
                emptyImage[i,j]=img[x,y]
    return emptyImage

img=cv2.imread("img/lenna.png")
zoom=function(img)
cv2.imshow("nearest difference",zoom)
cv2.imshow("image",img)
cv2.waitKey(0)


# 最邻近差值的缺点：放大后的图像有很严重的马赛克，缩小后的图像有很严重的 失真