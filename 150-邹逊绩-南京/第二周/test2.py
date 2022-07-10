#coding=utf8
import cv2

img = cv2.imread("lenna.png")
# cv2.imshow('', img)
imggray = img
imggray[:,:,0] = img[:,:,0] * 0.11 + img[:,:,1] * 0.59 + img[:,:,2] * 0.3
imggray[:,:,1] = imggray[:,:,0]
imggray[:,:,2] = imggray[:,:,0]
cv2.imshow('灰度化',imggray)
cv2.waitKey(0)

m1, n1 = imggray.shape[0:2]
for m in range(m1):
    for n in range(n1):
        a = imggray[m, n, 0]
        if a < 128:
            a = 255
        else:
            a = 0
        imggray[m, n, 0] = a
        imggray[m, n, 1] = a
        imggray[m, n, 2] = a

cv2.imshow('二值化',imggray)
cv2.waitKey(0)
