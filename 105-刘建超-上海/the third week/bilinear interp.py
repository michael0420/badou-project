# -*- coding:utf-8 -*-

# 双线性差值
import cv2
import numpy as np


def bilinear_interp(img, outdim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = outdim[1], outdim[0]
    print("src_h=,src_w=", src_h, src_w)
    print("dst_h=,dst_w=", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    emptyImage = np.zeros((dst_h, dst_w, channel), dtype=np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                temp1 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp2 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                emptyImage[dst_y, dst_x, i] = int((src_y1 - src_y) * temp1 + (src_y - src_y0) * temp2)

    return emptyImage


img = cv2.imread("lenna.png")
zoomimage = bilinear_interp(img, (800, 800))
print(zoomimage)
print(zoomimage.shape)
cv2.imshow("image show", img)
cv2.imshow("bilinear interpolation image show", zoomimage)
cv2.waitKey()
