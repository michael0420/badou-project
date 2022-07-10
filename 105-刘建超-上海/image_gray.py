# -*- coding: utf-8 -*-

# 彩色图像的灰度化、二值化
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 灰度化
img = cv2.imread("lenna.png", 1)
h, w = img.shape[:2]
image_gray = np.zeros([h, w], img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i, j]
        image_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)
print("image show gray: %s" % image_gray)
cv2.imshow("image show gray", image_gray)
cv2.waitKey(1000)

plt.subplot(2, 2, 1)
img = plt.imread("lenna.png")
plt.imshow(img)
print("***image lenna***")
print(img)

plt.subplot(2, 2, 2)
plt.imshow(image_gray, cmap='gray')
print("***image gray***")
print(image_gray)

# 二值化
plt.subplot(2, 2, 3)
rows, cols = image_gray.shape
for i in range(rows):
    for j in range(cols):
        if (image_gray[i, j] <= 128):
            image_gray[i, j] = 0
        else:
            image_gray[i, j] = 255
plt.imshow(image_gray, cmap='gray')
print("***image binary***")
print(image_gray)
plt.show()
