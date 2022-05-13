
import cv2
import numpy as np


# 灰度化
def function_gray(img):
    height, width, channels = img.shape
    b, g, r = cv2.split(img)
    # cv2.imshow("Blue 1", b)
    # cv2.imshow("Green 1", g)
    # cv2.imshow("Red 1", r)
    empty_image = np.zeros((height, width, channels), np.uint8)
    for i in range(height):
        for j in range(width):
            empty_image[i, j] = round(0.3*r[i, j]+0.59*g[i, j]+0.11*b[i, j])
    return empty_image


# 二值图
def function(img):
    height, width = img.shape[0:2]
    for i in range(height):
        for j in range(width):
            if img[i, j].mean()/255 <= 0.5:
                img[i, j] = 0
            else:
                img[i, j] = 255
    return img


img_src = cv2.imread("lenna.png")
cv2.imshow("lenna.png", img_src)

zoom = function_gray(img_src)
cv2.imshow("trans_gray", zoom)

zoom_2 = function(zoom)
cv2.imshow("Binary Image", zoom_2)

cv2.waitKey(0)

