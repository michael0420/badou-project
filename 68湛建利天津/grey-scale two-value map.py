import numpy as np
import cv2

#读取图片，显示行数、列数、通道数
img = cv2.imread('C:/Users/Administrator/Desktop/123.jpg')
cv2.namedWindow("image")
cv2.imshow("image", img)
rows = img.shape[0]
cols = img.shape[1]
channel = img.shape[2]
print("rows=", rows, " cols=", cols, " channel=", channel)

#创建空白单通道图像
greyimg = np.zeros((rows, cols), np.uint8)
greyimg.fill(0)
wbimg = np.zeros((rows, cols), np.uint8)
wbimg.fill(0)

#彩色图转灰度图
for i in range(rows):
    for j in range(cols):
        r = img[i, j, 0]
        g = img[i, j, 1]
        b = img[i, j, 2]
        greyimg[i, j] = r*0.3 + g*0.59 + b*0.11
cv2.namedWindow("greyimage")
cv2.imshow("greyimage", greyimg)

#灰度图像转二值图像
threshold = 150
for i in range(rows):
    for j in range(cols):
        value = greyimg[i, j]
        if value > threshold:
            wbimg[i, j] = 255
        else:
            wbimg[i, j] = 0
cv2.namedWindow("wbimage")
cv2.imshow("wbimage", wbimg)

cv2.waitKey(0)
cv2.destroyAllWindows()
