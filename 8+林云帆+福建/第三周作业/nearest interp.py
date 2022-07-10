import numpy as np
import cv2


def nearest_interp(img, size_1, size_2):
    H, W, C = img.shape
    empty_image = np.zeros((size_1, size_2, C), np.uint8)
    sh = size_1 / H
    sw = size_2 / W
    for i in range(size_1):
        for j in range(size_2):
            x = int(i / sh)
            y = int(j / sw)
            empty_image[i, j] = img[x, y]
    return empty_image


img = cv2.imread('lenna.png')
dst = nearest_interp(img,800,800)
print(dst)
print(dst.shape)
cv2.imshow('nearest interp',dst)
cv2.imshow('image',img)
cv2.waitKey(0)