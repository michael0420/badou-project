import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
'''
1.浮点算法 Gray = R0.3+G0.59+B0.11
2.整数算法 Gray = (R30+R59+B11)/100
3.移位算法 Gray = (R76+G151+B*28)>>8
4.平均算法 Gray = (R+G+B)/3
5.仅取绿色 Gray = G；
'''
#灰度化算法实现
def graydu():
    img = cv2.imread('dad.jpeg')
    h,w=img.shape[:2]
    print(h,w)#理解为h为行，w为列
    imggray = np.zeros([h,w],img.dtype)
    #遍历每一行每一列的值，再进行浮点化等操作(相当于对每一个像素点进行操作)
    for i in range(h):
        for j in range(w):
            m = img[i,j]
            imggray[i,j] = int(m[0]*0.11+m[1]*0.59+m[2]*0.3)#因为opencv读取的通道排列为BGR所以顺序不同
    print("image show gray: %s"%imggray)
    # cv2.imshow("image show gray",imggray)
    return imggray
# cv2.imshow("image show gray",graydu())
#二值化算法实现
def twozhi():
    img_gray = graydu()/255#除255 使其都在0，1范围内
    rows, cols = img_gray.shape
    for i in range(rows):
        for j in range(cols):
            if (img_gray[i, j] <= 0.5):
                img_gray[i, j] = 0
            else:
                img_gray[i, j] = 1
    return img_gray
#仅取绿色
def togreen():
    img = cv2.imread('dad.jpeg')
    h,w = img.shape[:2]
    imgreen = np.zeros([h,w],img.dtype)
    for i in range(h):
        for j in range(w):
            m = img[i,j]
            imgreen[i,j] = int(m[1])
    print("image show gray: %s" % imgreen)
    return imgreen
# cv2.imshow("image show gray1",togreen())
# cv2.waitKey(0)
plt.subplot(221)
img = plt.imread("dad.jpeg")
plt.imshow(img)
plt.subplot(222)
plt.imshow(graydu(), cmap='gray')##默认热力图，需要加上cmap='gray'
plt.subplot(223)
plt.imshow(twozhi(), cmap='gray')##默认热力图，需要加上cmap='gray'
plt.subplot(224)
plt.imshow(togreen(), cmap='gray')
plt.show()
'''
灰度化直接调用函数实现：
# 灰度化
img_gray = rgb2gray(img)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

二值化使用numpy实现
img_binary = np.where(img_gray >= 0.5, 1, 0)
'''