import cv2 as cv
import matplotlib.pyplot as plt
import  numpy as np

img=cv.imread('lenna.png',1)
grayimg=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#求灰度图的均衡化之后的图
grayequlize=cv.equalizeHist(grayimg)

#求彩色图像的均衡化后的图：rgb---》合并
r,g,b=cv.split(img)

rh=cv.equalizeHist(r)
gh=cv.equalizeHist(g)
bh=cv.equalizeHist(b)

imgh=cv.merge((rh,gh,bh))
cv.imshow('compare_gray',np.hstack([grayimg,grayequlize]))
cv.imshow('compare_scr',np.hstack([img,imgh]))

# cv.imshow('compare_src',np.hstack([grayimg,grayequlize]))

plt.figure('gray')
plt.hist(grayimg.ravel(),256,color='r')
plt.hist(grayequlize.ravel(),256,color='g')

plt.show()


cv.waitKey(0)
cv.destroyAllWindows()