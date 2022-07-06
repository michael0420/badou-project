# import
import numpy as np
import matplotlib.pyplot as plt


# plt.imread 导入图片
image = plt.imread('Image/lenna.png')
print("image_lenna:",image.shape)
print("image_type:",type(image))
print(image)
# subplot(131) 原图
plt.subplot(131)
plt.imshow(image)


# 灰度化
h,w = image.shape[:2]
image2gray = np.zeros((h,w),image.dtype)
print("image2gray:",image2gray.shape)
for i in range(h):
    for j in range(w):
        image2gray[i,j] = image[i,j][0]*0.3 + image[i,j][1]*0.59 + image[i,j][2]*0.11
# subplot(131) 灰度图
plt.subplot(132)
plt.imshow(image2gray,cmap='gray')


# 二值化
image2binary = np.zeros((h,w),image.dtype)
for i in range(h):
    for j in range(w):
        image2binary[i,j] = 0 if image2gray[i,j] < 0.4 else 1
# subplot(133) 二值图
plt.subplot(133)
plt.imshow(image2binary,cmap='gray')
plt.show()







