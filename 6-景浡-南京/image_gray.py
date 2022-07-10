"""

彩色图像的灰度化和二值化

"""
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.color import rgb2gray
# 灰度化


img = cv2.imread("2020_03_14.png")
h, w = img.shape[:2]  # 获取图像的高和宽
img_gray = np.zeros([h, w], img.dtype)  # 创建一张和当前图像大小一样的单通道图像
for i in range(h):
    for j in range(w):
        m = img[i, j]
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)  # 将BGR坐标转化为Gray坐标并赋值给新图像
print(img_gray)
print("image show gray: %s" % img_gray)
cv2.imshow("image show gray",img_gray)
cv2.waitKey(0)

#二值化
rows, cols = img_gray.shape
for i in range(rows):
    for j in range(cols):
        if(img_gray[i,j]<=125):
            img_gray[i,j] = 0
        else:
            img_gray[i,j] = 255
cv2.imshow("image show binary", img_gray)
cv2.waitKey(0)

#原图
plt.subplot(221)
# img = cv2.imread("lenna.png")     cv和plt的区别
img = plt.imread("2020_03_14.png")
plt.imshow(img)
print("---image lenna----")
print(img)

#灰度化
img_gray = rgb2gray(img)
# img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("---image gray----")
print(img_gray)

#二值化
img_binary = np.where(img_gray <=0.5, 0, 1)
print("-----imge_binary------")
print(img_binary)
print(img_binary.shape)

plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()