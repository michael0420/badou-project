import numpy as np
import skimage.color
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 输入并显示原始图像
# img =cv2.imread("lenna.png")
# cv2.imshow("图片展示", img)
# cv2.waitKey(0)
plt.subplot(131)
img = plt.imread("lenna.png")                    # plt读入的图片矩阵像素值归一化到了0-1之间
plt.imshow(img)
print(img)
h, w = img.shape[:2]

# 转换为灰度图
for i in range(h):
    for j in range(w):
        m = img[i][j]
        img[i][j] = m[0]*0.3 + m[1]*0.59 + m[2]*0.11
plt.subplot(132)
plt.imshow(img, cmap='gray')
img_mean = img.mean(axis=2)                     # 由于plt读入的图片默认为三通道，此代码把三通道聚合为1通道
print("-------------image gray----------------")
print(img_mean)

# 转换为二值图
plt.subplot(133)
a, b = img_mean.shape
print(a,b)
for i in range(a):
    for j in range(b):
        if img_mean[i][j] > 0.5:
            img_mean[i][j] = 1
        else:
            img_mean[i][j] = 0
plt.imshow(img_mean, cmap='gray')
plt.show()

