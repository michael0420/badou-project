import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r'Lenna.jpg')
# print(img.shape)
h, w = img.shape[:2]
print(h, w)
img_grey = np.zeros([h, w], img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i, j]
        print(m)
        img_grey[i, j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)


cv2.imshow('img', img_grey)
cv2.waitKey(0)

img_binary = np.where(img_grey >= 0.5, 1, 0)

cv2.imshow('img_bin', img_binary)
cv2.waitKey(0)
plt.subplot(223)
plt.imshow(img_binary)
plt.show()

