import cv2
import cv2 as cv
import numpy as np

img = cv2.imread("E:\\lenna.png")
cv2.imshow("img", img)

# 灰度图
rows, cols = img.shape[:2]
img_gray = np.zeros([rows, cols], img.dtype)
for i in range(rows):
    for j in range(cols):
        m = img[i][j]
        img_gray[i][j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)
cv2.imshow("img_gray", img_gray)

# 二值化
img_binary = img_gray
rows, cols = img_gray.shape
for i in range(rows):
    for j in range(cols):
        if img_gray[i][j] < 127:
            img_gray[i][j] = 0
        else:
            img_gray[i][j] = 255
cv2.imshow("img_binary", img_binary)

cv2.waitKey(0)
