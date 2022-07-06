# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
import numpy as np

def nearest_interp(img,dst_h,dst_w):
    src_h,src_w,channel=img.shape
    empty_img=np.zeros((dst_h,dst_w,channel),np.uint8)
    scale_x=dst_h/src_h
    scale_y=dst_w/src_w
    for i in range(dst_h):
        for j in range(dst_w):
            x=int(i/scale_x)
            y=int(j/scale_y)
            empty_img[i,j]=img[x,y]
    return empty_img





if __name__=='__main__':
    img=cv2.imread("D:\lenna.png")
    ans_img=nearest_interp(img,800,800)
    cv2.imshow("source",img)
    cv2.imshow("process",ans_img)
    cv2.waitKey(0)
