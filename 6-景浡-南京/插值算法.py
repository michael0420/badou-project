import cv2
import numpy as np

W=800
N=800
img = cv2.imread("lenna.png")
#最近邻插值
def function(img,W,N):
    height,width,channels = img.shape
    emptyImage = np.zeros((W,N,channels),np.uint8)
    sh = W/height
    sw = N/width
    for i in range(W):
        for j in range(N):
            x = int(i/sh+0.5)
            y = int(j/sw+0.5)
            emptyImage [i,j] = img[x,y]
    return emptyImage


zoom = function(img,W,N)
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp",zoom)
cv2.imshow("image",img)
cv2.waitKey(0)

#双线性插值
def bilinear_interpolation(img,out_dim):
    src_h,src_w,channels = img.shape
    dst_h,dst_w = out_dim[1], out_dim[0]
    print("scr_h scr_w = ",src_h,src_w)
    print("dst_h dst_w = ",dst_h,dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h,dst_w,3),dtype=np.uint8)
    scale_x,scale_y = float(src_w)/dst_w,float(src_h)/dst_h
    for i in range(3):
        for j in range(dst_h):
            for k in range(dst_w):
                #中心重叠
                src_x = (j+0.5)*scale_x - 0.5
                src_y = (k+0.5)*scale_y - 0.5
                #取整
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_h - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_w - 1)
                #公式
                temp0 = (src_x1 - src_x) * img[src_x0, src_y0, i] + (src_x - src_x0) * img[src_x1, src_y0, i]
                temp1 = (src_x1 - src_x) * img[src_x0, src_y1, i] + (src_x - src_x0) * img[src_x1, src_y1, i]
                dst_img[j,k,i]= int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    return  dst_img

dst = bilinear_interpolation(img,(W,N))
print(dst)
print(dst.shape)
cv2.imshow("bilinear interp",dst)
cv2.imshow("image",img)
cv2.waitKey(0)