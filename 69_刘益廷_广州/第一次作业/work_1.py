"""
work_1:  实现模块功能：彩色图像的灰度化&二值化
运用包/模块：PIL、matplotlib.pyplot、numpy、cv2
"""
from PIL import Image   #from（从） 包/模块 import（导入） 模块/方法
import matplotlib.pyplot as plt   ## as （别名模块）
import numpy as np
import cv2

#灰度化
# img = cv2.imread("sunset.jpg")          #cv2读取的是BGR
img = plt.imread("sunset.jpg")
h,w = img.shape[:2]                     #获取img的hide和wide；img.shape：(行，列，通道）
img_gray = np.zeros([h,w],img.dtype)    #创建跟当前图片大小一样的单通道图像
for i in range(h):
    for j in range(w):
        m = img[i,j]                    #取出当前high和wide中的R、G、B坐标
        # print(m)
        #img_gray[i,j] = int(m[0]*11 + m[1]*59 + m[2]*30)  #加权平均法：f(i,j)=0.30R(i,j)+0.59G(i,j)+0.11B(i,j))
        img_gray[i,j] = int((m[0]+m[1]+m[2])) /3           #平均值法： f(i,j)=(R(i,j)+G(i,j)+B(i,j))/3
# print(img_gray)
# print("image show gray: %s"%img_gray)
# cv2.namedWindow('image show gray',0)
# cv2.resizeWindow('image show gray', 640, 480)
# cv2.imshow('image show gray',img_gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

plt.subplot(221)
plt.imshow(img)
print("---image sunset----")
print(img)
plt.subplot(222)
plt.imshow(img_gray,cmap='gray')
print("---image sunset gray----")
print(img_gray)

#二值化
rows, cols = img_gray.shape
img_binary = np.zeros([rows,cols],img.dtype)    #创建跟当前图片大小一样的单通道图像
for i in range(rows):
    for j in range(cols):
        if (img_gray[i, j] <= 50):
            img_binary[i, j] = 0
        else:
            img_binary[i, j] = 1

# img_binary = np.where(img_gray <= 0.3, 1, 0)
print("-----imge_binary------")
print(img_binary)
print(img_binary.shape)
plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()
