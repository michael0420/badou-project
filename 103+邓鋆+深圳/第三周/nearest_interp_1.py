import numpy as np
import cv2

def interp(img,dsize):
    w,h,c = img.shape
    print(w,h,c )
    newImg = np.zeros((dsize[0],dsize[1],c),np.uint8)
    nw = dsize[0]/w
    nh = dsize[1]/h

    for i in range(dsize[0]):
        for j in range(dsize[1]):
            xw = round(i/nw)
            if xw >= w:  #避免舍入后 超出原图片大小
                xw = w - 1
            xh = round(j/nh)
            if xh  >= h:
                xh = h - 1
            newImg[i,j] = img[xw,xh]

    return newImg


img = cv2.imread('w2.png')
zoom = interp(img,[int(img.shape[0]*0.6),int(img.shape[1]*0.6)]) # 缩小
cv2.imshow("image",zoom)

zoom2 = interp(img,[int(img.shape[0]*1.5),int(img.shape[1]*1.5)]) # 放大
cv2.imshow("image2",zoom2)
cv2.imshow("img",img)
cv2.waitKey(0)