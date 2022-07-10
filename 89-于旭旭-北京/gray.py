from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from cv2 import THRESH_BINARY
import pylab

# 灰度化
plt.subplot(221)
img=cv2.imread("E:\yuxuxu\lenna.png")
img_gray=rgb2gray(img)
plt.imshow(img_gray,cmap='gray')

###二值化
plt.subplot(222)
img=cv2.imread("E:\yuxuxu\lenna.png")
img_gray=rgb2gray(img)
img_binary=np.where(img_gray>=0.5,1,0)
plt.imshow(img_binary,cmap='gray')
plt.show()

##灰度化：
img=cv2.imread("E:\yuxuxu\lenna.png")
h,w=img.shape[:2]  ##获取图片的高度与宽度 [:2] 表示取列表的第0个、第1个元素，不包含第二个元素
img_gray=np.zeros([h,w],img.dtype) ##取h行，w列的矩阵，元素都是0，dtype：数据类型为img的类型
for i in range(h):
    for j in range(w):
        m=img[i,j]
        img_gray[i,j]=int(m[0]*0.11+m[1]*0.59+m[2]*0.3)
print(img_gray)
print("image show gray: %s"%img_gray)
cv2.imshow("image show gray",img_gray)

plt.subplot(221)
img=plt.imread("E:\yuxuxu\lenna.png")
plt.imshow(img)
print(img)

###灰度化
img_gray=rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray,cmap='gray')


###二值化
img_binary=np.where(img_gray>=0.5,1,0)
plt.subplot(223)
plt.imshow(img_binary,cmap='gray')
plt.show()
