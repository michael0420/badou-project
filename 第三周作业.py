import cv2
import numpy as np
from matplotlib import pyplot as plt
#1.临近插值
def function(img):
    height,width,channels =img.shape
    emptyImage=np.zeros((800,800,channels),np.uint8)
    sh=800/height
    sw=800/width
    for i in range(800):
        for j in range(800):
            x=int(i/sh)
            y=int(j/sw)
            emptyImage[i,j]=img[x,y]
    return emptyImage

img=cv2.imread("jjy.jpg")
zoom=function(img)
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp",zoom)

#2.双线性插值
def bilinear_interpolation(img,out_dim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]
    print ("src_h, src_w = ", src_h, src_w)
    print ("dst_h, dst_w = ", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h,dst_w,3),dtype=np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                src_x = (dst_x + 0.5) * scale_x-0.5
                src_y = (dst_y + 0.5) * scale_y-0.5
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1 ,src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)
                temp0 = (src_x1 - src_x) * img[src_y0,src_x0,i] + (src_x - src_x0) * img[src_y0,src_x1,i]
                temp1 = (src_x1 - src_x) * img[src_y1,src_x0,i] + (src_x - src_x0) * img[src_y1,src_x1,i]
                dst_img[dst_y,dst_x,i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
        return dst_img
if __name__ == '__main__':
    img = cv2.imread('jjy.jpg')
    dst = bilinear_interpolation(img,(700,700))
    cv2.imshow('bilinear interp',dst)
    cv2.waitKey()
'''
3.中心重叠推导
（X1,Y1）--->(X2,Y2)
比例p=X1/X2;
原来中心（X1,Y1)/2, 转换后中心或理想中心（X2,Y2）/2
设图一像素点个数为M*M,图二像素点个数为N*N
分类讨论：
若N是素数即奇数，那么中心就是(N+1/2  ,N+1/2 )
若N是偶数，那么中心就是（N/2  +0.5,N/2  +0.5）
因为p=M/N
所以，(N+1)/2=(M +P)/2P
即N+0.5=0.5(M/P)+0.5
'''
#4.图像均衡化

#灰度均衡
img = cv2.imread("jjy.jpg", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow("image_gray", gray)
dst = cv2.equalizeHist(gray)
# 直方图
hist = cv2.calcHist([dst],[0],None,[256],[0,256])
plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()
cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)

#彩色均衡
img = cv2.imread("jjy.jpg", 1)
cv2.imshow("src", img)
# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((bH, gH, rH))
cv2.imshow("dst_rgb", result)
cv2.waitKey(0)
