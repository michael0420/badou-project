#修改：实现了四舍五入以及将赋值改到了函数外
#最邻近插值算法（上采样）完成：放大不会更糊。
import cv2
import numpy as np
def function(img,aim_h,aim_w):
    height,width,channels =img.shape
    emptyImage=np.zeros((aim_h,aim_w,channels),np.uint8)     #size
    sh=aim_h/height
    sw=aim_w/width                                         #proportion
    for i in range(aim_h):
        for j in range(aim_w):
            x=int(i/sh+0.5)
            y=int(j/sw+0.5)                                  #Determines adjacent pixel values
            emptyImage[i,j]=img[x,y]
    return emptyImage

aim_h = 600
aim_w = 600
img=cv2.imread("lenna.png")     #Read the image，cv2.imread()得到的img数据类型是np.array()类型。
zoom=function(img,aim_h,aim_w)
print(zoom)
print(zoom.shape)
print(img.shape)
cv2.imshow("nearest interp",zoom)       #image display （窗口号，显示图像）
cv2.imshow("image",img)
cv2.waitKey(0)                          #waitkey控制着imshow的持续时间

