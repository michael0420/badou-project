# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
import numpy as np

def bilinear_interpolation(img,dst_h,dst_w):
    src_h,src_w,channel=img.shape
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    empty_img=np.zeros((dst_h,dst_w,channel),np.uint8)
    scale_x=dst_h/src_h
    scale_y=dst_w/src_w
    for k in range(channel):    
        for i in range(dst_h):
            for j in range(dst_w):
                # 找到对应点，中心重叠，以保证任一点均可在原图中找到对应点
                src_x=(i+0.5)/scale_x-0.5
                src_y=(j+0.5)/scale_y-0.5
                # 双线性插值，三次插值算法
                src_x0=int(np.floor(src_x))
                src_x1=min(src_x0+1,src_h-1)
                src_y0=int(np.floor(src_y))
                src_y1=min(src_y0+1,src_w-1)
                temp0 = (src_x1 - src_x) * img[src_y0,src_x0,k] + (src_x - src_x0) * img[src_y0,src_x1,k]
                temp1 = (src_x1 - src_x) * img[src_y1,src_x0,k] + (src_x - src_x0) * img[src_y1,src_x1,k]
                empty_img[j,i,k] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)                
    return empty_img





if __name__=='__main__':
    img=cv2.imread("D:\lenna.png")
    ans_img=bilinear_interpolation(img,700,700)
    cv2.imshow("source",img)
    cv2.imshow("process",ans_img)
    cv2.waitKey(0)
