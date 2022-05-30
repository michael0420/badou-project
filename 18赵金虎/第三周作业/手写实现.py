import cv2 as cv
import numpy as np

# 最近领差值
def nearest_interp(img,fx,fy):
    height,width,channels =img.shape
    height_new,width_new = int(fx*height),int(fy*width)
    emptyImage = np.zeros((height_new,width_new,channels),np.uint8)
    sh,sw = fx,fy
    for i in range(height_new):
        for j in range(width_new):
            x = int(i/sh)
            y = int(j/sw)
            emptyImage[i,j] = img[x,y]
    return emptyImage

# 双线性差值
def bilinear_interp(img,fx,fy):
    src_h,src_w,channel = img.shape
    dst_h,dst_w = int(fx * src_h), int(fy * src_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # find the origin x and y coordinates of dst image x and y
                # use geometric center symmetry
                # if use direct way, src_x = dst_x * scale_x
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                # find the coordinates of the points which will be used to compute the interpolation
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # calculate the interpolation
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    return dst_img

img = cv.imread('lenna.png')
res = nearest_interp(img,1.2,1.2)
cv.imshow('nearest_interp',res)
res1 = bilinear_interp(img,1.2,1.2)
cv.imshow('Bilinear_interp',res1)
cv.waitKey(0)