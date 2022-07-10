# -*- coding: utf-8 -*-
"""
Created on Sat May 14 19:44:33 2022

@author: Administrator
"""
# %%
import matplotlib.pyplot as plt
import numpy as np
import cv2

# %% 最邻近插值
def nearest(img, dst_w, dst_h):
    src_h, src_w, channel =img.shape
    #dst_h, dst_w = out_dim[1], out_dim[0]
    print ("src_h, src_w = ", src_h, src_w)
    print ("dst_h, dst_w = ", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h,dst_w,channel),dtype=np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(dst_h):
        for j in range(dst_w):
            x=int(i*scale_x+0.5)  
            y=int(j*scale_y+0.5)
            dst_img[i,j]=img[x,y]
    return dst_img

# %% 双线性插值
def bilinear_interpolation(img, dst_w, dst_h):
    src_h, src_w, channel = img.shape
    #dst_h, dst_w = out_dim[1], out_dim[0]
    print ("src_h, src_w = ", src_h, src_w)
    print ("dst_h, dst_w = ", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h,dst_w,channel),dtype=np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
 
                # find the origin x and y coordinates of dst image x and y
                # use geometric center symmetry
                # if use direct way, src_x = dst_x * scale_x
                src_x = (dst_x + 0.5) * scale_x-0.5
                src_y = (dst_y + 0.5) * scale_y-0.5
 
                # find the coordinates of the points which will be used to compute the interpolation
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1 ,src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)
 
                # calculate the interpolation
                temp0 = (src_x1 - src_x) * img[src_y0,src_x0,i] + (src_x - src_x0) * img[src_y0,src_x1,i]
                temp1 = (src_x1 - src_x) * img[src_y1,src_x0,i] + (src_x - src_x0) * img[src_y1,src_x1,i]
                dst_img[dst_y,dst_x,i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
 
    return dst_img

# %% 中心重叠
def middle_interpolate(img, dst_w, dst_h):
    src_h, src_w, channel =img.shape
    #dst_h, dst_w = out_dim[1], out_dim[0]
    print ("src_h, src_w = ", src_h, src_w)
    print ("dst_h, dst_w = ", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h,dst_w,channel),dtype=np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for dst_y in range(dst_h):
        for dst_x in range(dst_w):
            src_x = int((dst_x + 0.5) * scale_x-0.5)
            src_y = int((dst_y + 0.5) * scale_y-0.5)
            dst_img[dst_y,dst_x] = img[src_y,src_x]
                
    return dst_img


# %% 直方图均衡化
def equlization(img):
    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    result = cv2.merge((bH, gH, rH))
    return result
    
# %% test functions
if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    inputs = int(input("最邻近插值输入1\n双线性插值输入2\n中心重叠输入3\n直方图均衡化输入4:\n"))
    if inputs == 1:
        inputs = input("请输入放大或缩小后的宽高 空格隔开:\n")
        dst=nearest(img,int(inputs.split(' ')[0]),int(inputs.split(' ')[1]))
        print(dst.shape)
        cv2.imshow("nearest interp",dst)
        cv2.imshow("image",img)
        cv2.waitKey(0)
    if inputs == 2:
        inputs = input("请输入放大或缩小后的宽高 空格隔开:\n")
        dst = bilinear_interpolation(img, int(inputs.split(' ')[0]), int(inputs.split(' ')[1]))
        cv2.imshow('bilinear interp', dst)
        cv2.imshow('img', img)
        cv2.waitKey()
    if inputs == 3:
        inputs = input("请输入放大或缩小后的宽高 空格隔开:\n")
        dst = middle_interpolate(img, int(inputs.split(' ')[0]), int(inputs.split(' ')[1]))
        cv2.imshow('middle_interpolate', dst)
        cv2.imshow('img', img)
        cv2.waitKey()
    if inputs == 4:
        dst = equlization(img)
        cv2.imshow('equlization', dst)
        cv2.imshow('img', img)
        cv2.waitKey()