# 双线性， the bilinear interpolation

import cv2
import numpy as np

# scale: src/dst
def bilinearInterp(img, dst_h, dst_w):
    src_h, src_w, channel = img.shape
    targetImage = np.zeros((dst_h, dst_w, channel), np.uint8)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    h = float(src_h/dst_h)
    w = float(src_w/dst_w)
    for i in range(channel):
        for dst_y in range(dst_h):
            src_y = (dst_y + 0.5) * h - 0.5
            if src_y <= 0:
                src_y = 0
                src_y0 = 0
                src_y1 = 0
            elif src_y >= src_h-1:
                src_y = src_h-1
                src_y0 = src_h-1
                src_y1 = src_h-1
            else:
                src_y0 = int(np.floor(src_y))
                src_y1 = src_y0 + 1
            for dst_x in range(dst_w):
                src_x = (dst_x + 0.5) * w - 0.5
                if src_x <= 0:
                    src_x = 0
                    src_x0 = 0
                    src_x1 = 0
                elif src_x >= src_w-1:
                    src_x = src_w-1
                    src_x0 = src_w-1
                    src_x1 = src_w-1
                else:
                    src_x0 = int(np.floor(src_x))
                    src_x1 = src_x0 + 1
                temp0 = (src_x1 - src_x) * img[src_y0,src_x0,i] + (src_x - src_x0) * img[src_y0,src_x1,i]
                temp1 = (src_x1 - src_x) * img[src_y1,src_x0,i] + (src_x - src_x0) * img[src_y1,src_x1,i]
                targetImage[dst_y,dst_x,i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    return targetImage


img = cv2.imread("lenna.png")
target = bilinearInterp(img,700,700)
# print(target)
print(target.shape)
cv2.imshow("bilinear interp",target)
# cv2.imshow("image",img)
cv2.waitKey(0)


# scale: (src-1)/(dst-1)
def bilinearInterp2(img, dst_h, dst_w):
    src_h, src_w, channel = img.shape
    targetImage = np.zeros((dst_h, dst_w, channel), np.uint8)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    h = float((src_h-1)/(dst_h-1))
    w = float((src_w-1)/(dst_w-1))
    for i in range(channel):
        for dst_y in range(dst_h):
            src_y = (dst_y) * h
            if src_y <= 0:
                src_y = 0
                src_y0 = 0
                src_y1 = 0
            elif src_y >= src_h-1:
                src_y = src_h-1
                src_y0 = src_h-1
                src_y1 = src_h-1
            else:
                src_y0 = int(np.floor(src_y))
                src_y1 = src_y0 + 1
            for dst_x in range(dst_w):
                src_x = (dst_x) * w
                if src_x <= 0:
                    src_x = 0
                    src_x0 = 0
                    src_x1 = 0
                elif src_x >= src_w-1:
                    src_x = src_w-1
                    src_x0 = src_w-1
                    src_x1 = src_w-1
                else:
                    src_x0 = int(np.floor(src_x))
                    src_x1 = src_x0 + 1
                temp0 = (src_x1 - src_x) * img[src_y0,src_x0,i] + (src_x - src_x0) * img[src_y0,src_x1,i]
                temp1 = (src_x1 - src_x) * img[src_y1,src_x0,i] + (src_x - src_x0) * img[src_y1,src_x1,i]
                targetImage[dst_y,dst_x,i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    return targetImage


img = cv2.imread("lenna.png")
target = bilinearInterp2(img,700,700)
# print(target)
print(target.shape)
cv2.imshow("bilinear interp2",target)
# cv2.imshow("image",img)
cv2.waitKey(0)