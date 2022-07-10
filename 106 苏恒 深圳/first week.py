import cv2
import matplotlib.pyplot as plt
import numpy as np

#
img=cv2.imread('C:\\Users\\Holl\\Desktop\\log\\lenna.png')
cv2.imshow('img1',img)

#灰度化
img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img_gray=cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)
# cv2.imshow('img2',img_gray)
#
#二值化
ret,dst=cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
# cv2.imshow('img3',dst)
all=np.hstack((img_gray,dst))
cv2.imshow('Gray and bw',all)

# 灰度化（方法二）
rows,cols=img.shape[0:2]
img_gray2=np.zeros([rows,cols],img.dtype)
for i in range(rows):
    for j in range(cols):
        m=img[i,j]
        img_gray2[i,j]=int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)

cv2.imshow('Method2',img_gray2)

#二值化(法二)
img_gray3=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
rows, cols = img_gray3.shape[0:2]
for i in range(rows):
    for j in range (cols):
        if img_gray3[i,j]<127:
            img_gray3[i,j]=0
        else:
            img_gray3[i,j]=255
cv2.imshow('bw2',img_gray3)



cv2.waitKey(0)
cv2.destroyWindow( )


