import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from skimage.color import rgb2gray

#灰度化(法1)
img = cv2.imread("C:\\Users\\LENOVO\\Desktop\\lenna.png")
high, wide = img.shape[:2]
img_gray = np.zeros([high, wide], img.dtype) #创建一张和当前图片相同大小的图片

for i in range(high):                           #遍历图像中的每一个点
    for j in range(wide):
        m = img[i, j]
        img_gray[i, j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)    #正常的工程中一般不用写这4行直接调用opencv接口就可以了
print(img_gray)
cv2.imshow("image show gray", img_gray)
#cv2.waitKey(100)

plt.subplot(221)                                                #分成2行2列的矩阵此处占一
img = plt.imread("C:\\Users\\LENOVO\\Desktop\\lenna.png")
plt.imshow(img)                                                 #设置绘制的图像
print("---img lenna---")
print(img)


#灰度法（法2）

#img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img_gray = img
img_gray = rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')               #设置绘制图像，设置绘制格式
print("---image gray----")
print(img_gray)


#二值化（法1）
rows, cols = img_gray.shape
for i in range(rows):
    for j in range(cols):
        if (img_gray[i, j]<0.5):
            img_gray[i, j] = 0
        else:
            img_gray[i, j] = 1
        img_binary=img_gray


plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
'''
img_binary = np.where(img_gray >= 0.5 , 1 ,0)

plt.subplot(223)
plt.imshow(img_binary, cmap = 'gray')
'''

plt.show()