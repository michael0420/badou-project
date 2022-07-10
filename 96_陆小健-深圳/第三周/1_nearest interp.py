# 最邻近插值法处理图像，在最邻近点改变像素，达到改变图像的效果，缺点是当放大图像时出现锯齿状现象
# 把图像改变成1000*1000时的效果
import cv2
import numpy as np
def function(img):
    height,width,channels =img.shape
    emptyImage=np.zeros((1000,1000,channels),np.uint8)
    sh=1000/height
    sw=1000/width
    # 运用int的小技巧
    for i in range(1000):
        for j in range(1000):
            x=int(i/sh)  
            y=int(j/sw)
            emptyImage[i,j]=img[x,y]
    return emptyImage

img=cv2.imread("lenna.png")
zoom=function(img)
# print(zoom)
print('img.shape = ',img.shape)
print('zoom.shape = ',zoom.shape)
cv2.imshow("nearest interp",zoom)
cv2.imshow("image",img)
cv2.waitKey(0)


