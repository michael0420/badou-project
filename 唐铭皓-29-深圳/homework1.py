import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


# 灰度化
img = cv.imread("E:/data_set/dog_data/train/dogs/dog.1.jpg")
h, w = img.shape[:2]
img_gray = np.zeros([h,w], img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i, j]
        img_gray[i,j] = m[2] * 0.3 + m[1] * 0.59 + m[1] * 0.11
print(img_gray)
print("img show gray: %s"%img_gray)
cv.imshow("img show gray", img_gray)

plt.subplot(221)
img = plt.imread("E:/data_set/dog_data/train/dogs/dog.1.jpg")
plt.imshow(img)
plt.title('RGB')

img_gray_1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
plt.subplot(222)
plt.imshow(img_gray_1)
plt.title('GRAY')

# 二值化
for i in range(h):
    for j in range(w):
        if img_gray[i, j] >= 128:
            img_gray[i, j] = 1
        else:
            img_gray[i,j] = 0
plt.subplot(223)
plt.imshow(img_gray)
plt.title('BINARY1')
img_binary = np.where(img_gray_1 >= 128, 1, 0)
plt.subplot(224)
plt.imshow(img_binary)
plt.title('BINARY2')

plt.show()
