import numpy as np
import matplotlib.pyplot as plt

image_original = plt.imread('lenna.png')  # 读取原始图片
h, w = image_original.shape[:2]  # 获取图片的高度和宽度
print("-----image_original------")
print(image_original)
plt.subplot(221)  # 在2行2列的第一个位置显示图片
plt.imshow(image_original)  # 显示原始图片


# 原始图片灰度化处理
image_gray = np.zeros([h, w], image_original.dtype)
for i in range(h):
    for j in range(w):
        image_temp = image_original[i, j]
        image_gray[i, j] = image_temp[0]*0.3 + image_temp[1]*0.59 + image_temp[2]*0.11  # RGB转化为Gray
print("-----image_gray------")
print(image_gray)
plt.subplot(222)
plt.imshow(image_gray, cmap='gray')


# 原始图片二值化处理
h1, w1 = image_gray.shape[:2]
image_binary = np.zeros([h1, w1], image_gray.dtype)
for i in range(h1):
    for j in range(w1):
        if image_gray[i, j] <= 0.5:
            image_binary[i, j] = 0
        else:
            image_binary[i, j] = 1
print("-----image_binary------")
print(image_binary)
plt.subplot(223)
plt.imshow(image_binary, cmap='gray')
plt.show()