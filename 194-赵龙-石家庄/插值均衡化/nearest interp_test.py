
import cv2
import numpy as np
def function(img):
    height,width,channels =img.shape
    print(height,width,channels)
    emptyImage=np.zeros((600,600,channels),np.uint8)
    sh=800/height
    sw=800/width
    for i in range(600):
        for j in range(600):
            x=int(i/sh)  
            y=int(j/sw)
            emptyImage[i,j]=img[x,y]
    return emptyImage

for i in range(10):
    print(i)

img=cv2.imread("lenna.png")
zoom=function(img)
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp",zoom)
cv2.imshow("image",img)
cv2.waitKey(0)


