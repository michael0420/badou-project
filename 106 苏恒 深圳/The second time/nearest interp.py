import cv2
import numpy as np
"""最邻近插值"""

def function(img,t):
    height,width,channels =img.shape
    emptyImage=np.zeros((t,t,channels),np.uint8)
    sh=t/height
    sw=t/width
    for i in range(t):
        for j in range(t):
            x=int(i/sh)  
            y=int(j/sw)
            emptyImage[i,j]=img[x,y]
    return emptyImage


if __name__ == '__main__':
    img=cv2.imread("lenna.png")
    print(img.shape)
    #放大到800
    test1=function(img,800)
    #缩小到256
    test2=function(img,256)
    cv2.imshow("image", img)
    cv2.imshow("nearest interp1",test1)
    cv2.imshow("nearest interp2", test2)
    cv2.waitKey(0)


