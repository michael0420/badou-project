"""
work_2_2:  实现双线性插值
运用包/模块：PIL、matplotlib.pyplot、numpy、cv2
"""
import cv2
import numpy as np

def function_bilinearinterp(img,out_dim):
    i_ww,i_hh,i_c = img.shape
    o_ww,o_hh = out_dim[0],out_dim[1]
    print("i_ww,i_hh = ",i_ww,i_hh)
    print("o_ww,o_hh = ",o_ww,o_hh)

    emptyImage = np.zeros((o_ww,o_hh,3),dtype=np.uint8)
    sw = float(i_ww)/o_ww
    sh = float(i_hh)/o_hh
    for k in range(3):
        for i in range(o_ww):
            for j in range(o_hh):
                #几何中心重合
                i_w = (i + 0.5) * sw - 0.5
                i_h = (j + 0.5) * sh - 0.5
                #取端坐标
                i_w0 = int(np.floor(i_w))
                i_w1 = int(min(i_w0 + 1, i_ww - 1))
                i_h0 = int(np.floor(i_h))
                i_h1 = int(min(i_h0 + 1, i_hh - 1))
                # if i_w0<0: i_w0 = i_w0+1
                # if i_w1<0: i_w1 = i_w1+1
                # if i_h0<0: i_h0 = i_h0+1
                # if i_h1<0: i_h1 = i_h1+1
                #计算插值
                temp0 = (i_w1 - i_w) * img[i_w0,i_h0,k] + (i_w - i_w0) * img[i_w1,i_h0,k]
                temp1 = (i_w1 - i_w) * img[i_w0,i_h1,k] + (i_w - i_w0) * img[i_w1,i_h1,k]
                emptyImage[i,j,k] = int((i_h1 - i_h) * temp0 + (i_h - i_h0) * temp1)
    return  emptyImage

# img = cv2.imread("lenna.png")
img = cv2.imread("sunset.jpg")
zoom = function_bilinearinterp(img,[800,1200])
print(zoom)
print(zoom.shape)
cv2.imshow("binearest interp",zoom)
cv2.imshow("original image",img)
cv2.waitKey(0)