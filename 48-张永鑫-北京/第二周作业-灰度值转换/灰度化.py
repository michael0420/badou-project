
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 灰度化
img = cv2.imread("lenna.png")
h, w = img.shape[:2]
img_gray = np.zeros([h, w], img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i, j]
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)

plt.subplot(221)
img = plt.imread("lenna.png")
plt.imshow(img)
print("---image lenna----")
print(img)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("---image gray----")
print(img_gray)

# 二值化
h, w = img.shape[:2]
img_binary = np.zeros([h, w], img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i, j]
        n = m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3
        if (n <= 0.5):
            img_binary[i, j] = 0
        else:
            img_binary[i, j] = 1

print("-----imge_binary------")
print(img_binary)
print(img_binary.shape)
plt.subplot(223)
plt.imshow(img_binary, cmap='binary')
plt.show()