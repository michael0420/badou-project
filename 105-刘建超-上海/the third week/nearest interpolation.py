# -*- coding: utf-8 -*-

# 最邻近差值
import cv2
import numpy as np


def nearest_interp(img, outdim):
    src_h, src_w, channel = img.shape[:3]
    dst_h, dst_w = outdim[1], outdim[0]
    print("src_h= %s" % src_h)
    print("src_w= %s" % src_w)
    print("dst_h= %s" % dst_h)
    print("dst_w= %s" % dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    emptyImage = np.zeros([dst_h, dst_w, channel], dtype=np.uint8)
    sh = dst_h / src_h
    sw = dst_w / src_w
    for i in range(dst_h):
        for j in range(dst_w):
            x = int(i / sh + 0.5)
            y = int(j / sw + 0.5)
            emptyImage[i, j] = img[x, y]
    return emptyImage


img = cv2.imread("lenna.png")
zoomImage = nearest_interp(img, [800, 800])
print(zoomImage)
print(zoomImage.shape)
cv2.imshow("nearest interpolation image show", zoomImage)
cv2.imshow("image show", img)
cv2.waitKey(0)
