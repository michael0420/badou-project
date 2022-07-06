import math

import cv2
import numpy as np
def function(img,zoomh,zoomw):
    height,width,channels =img.shape
    emptyImage=np.zeros((zoomh,zoomw,channels),np.uint8)
    sh=zoomh/height
    sw=zoomw/width
    for i in range(zoomh):
        for j in range(zoomw):
            x=int(i/sh)
            y=int(j/sw)
            out = near(i/sh,j/sw,x,y)
            emptyImage[i,j]=img[out[0],out[1]]
    return emptyImage

## check >0.5 math.cell or math.floor
def near(u,v,x,y):
    if( math.fabs(u - x) <0.5 and math.fabs(v - y) < 0.5) :
        return (math.floor(u),math.floor(v))
    if (math.fabs(u - x) >= 0.5 and math.fabs(v - y) < 0.5):
        return (math.ceil(u), math.floor(v))
    if (math.fabs(u - x) < 0.5 and math.fabs(v - y) >= 0.5):
        return (math.floor(u), math.ceil(v))
    if (math.fabs(u - x) >= 0.5 and math.fabs(v - y) >= 0.5):
        return (math.ceil(u), math.ceil(v))
   # if (u >= 0.5 and v < 0.5):
   #     return (i+1, j)
   # if (u < 0.5 and v >= 0.5):
   #    return (i, j+1)
   # if (u >= 0.5 and v >= 0.5):
   #     return (i+1, j+1)

img=cv2.imread("lenna.png")
zoomh = 800
zoomw = 900
zoom=function(img,zoomh,zoomw)
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp",zoom)
cv2.imshow("image",img)
cv2.waitKey(0)



