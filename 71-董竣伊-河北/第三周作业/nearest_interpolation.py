import cv2
import numpy as np


def zoomImage(sourceImage, dest_h, dest_w):
    src_h, src_w, channels = sourceImage.shape
    destImage = np.zeros((800, 800, channels), np.uint8)
    h_ratio = dest_h / src_h
    w_ratio = dest_w / src_w
    for i in range(dest_h):
        for j in range(dest_w):
            x = round(i / h_ratio)
            y = round(j / w_ratio)
            destImage[i, j] = sourceImage[x, y]
    return destImage


srcImage = cv2.imread("lenna.png")
destImage = zoomImage(srcImage, 800, 800)
print(destImage)
print(destImage.shape)
cv2.imshow("nearest interpolation", destImage)
cv2.imshow("source image", srcImage)
cv2.waitKey(0)
