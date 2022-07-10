import cv2
import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

# 通过cv2读取图片信息，BGR顺序
image = cv2.imread('lenna.png')
print("-" * 10 + "image" + "-" * 10)
print(image.dtype)
#print(image)
# cv2.namedWindow('image',cv2.WINDOW_AUTOSIZE)
# cv2.imshow('image rgb',image) # BGR输入，BGR输出
# cv2.waitKey(0)
# cv2.destroyAllWindows


# 将BGR转换为RGB顺序
image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
print("-" * 10 + "image rgb" + "-" * 10)
print(image_rgb.dtype)
#print(image_rgb)
# cv2.imshow('image',image_rgb) # BGR输入，RGB输出
# cv2.waitKey(0)


# image顺序为BGR，转换顺序为RGB，即反顺序
image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
print("-" * 10 + "image gray" + "-" * 10)
print(image_gray.dtype)
print(image_gray)
# cv2.imshow('image gray', image_gray)
# cv2.waitKey(0)


# 通过两个for循环按RGB顺序转换为灰度图，正顺序；int向下取整会产生一定误差
#print(image.shape)
h,w = image.shape[:2]
image_gray_rgb = np.zeros([h,w],np.float64)
for i in range(h):
    for j in range(w):
        image_gray_rgb[i,j] = (0.11*image[i][j][0] + 0.59*image[i][j][1] + 0.3*image[i][j][2])
print("-" * 10 + "image gray rgb" + "-" * 10)
print(image_gray_rgb.dtype)
print(image_gray_rgb)
# cv2.imshow('image gray rgb',image_gray_rgb)
# cv2.waitKey(0)


# image顺序为BGR，转换顺序为RGB，即反顺序，值的范围为0-1
img_gray = rgb2gray(image)
print("-" * 10 + "img gray" + "-" * 10)
print(img_gray.dtype)
print(img_gray)
# cv2.imshow('img gray', img_gray)
# cv2.waitKey(0)


# image_rgb顺序为RGB，转换顺序为RGB，值的范围为0-1
img_gray_rgb = rgb2gray(image_rgb)
print("-" * 10 + "img gray rgb" + "-" * 10)
print(img_gray_rgb.dtype)
print(img_gray_rgb)
# cv2.imshow('img gray rgb', img_gray_rgb)
# cv2.waitKey(0)


# 通过两个for循环转换为二值图
image_binary = np.zeros([h,w], image_gray.dtype)
for i in range(h):
    for j in range(w):
        if image_gray[i][j] > 127:
            image_binary[i,j] = 255
        else:
            image_binary[i,j] = 0
print("-" * 10 + "image binary" + "-" * 10)
print(image_binary.dtype)
print(image_binary)
#print(image_binary.dtype)
# cv2.imshow('image binary', image_binary)
# cv2.waitKey(0)


# 通过两个for循环转换为二值图
image_binary_rgb = np.zeros([h,w], image_gray.dtype)
for i in range(h):
    for j in range(w):
        if image_gray_rgb[i][j] >= 127.5:
            image_binary_rgb[i,j] = 255
        else:
            image_binary_rgb[i,j] = 0
print("-" * 10 + "image binary rgb" + "-" * 10)
print(image_binary_rgb.dtype)
print(image_binary_rgb)
# cv2.imshow('image binary rgb', image_binary_rgb)
# cv2.waitKey(0)


# 通过numpy转换为二值图
img_binary = np.where(img_gray >= 0.5, 1, 0) 
img_binary = img_binary.astype(np.float64)
print("-" * 10 + "img binary" + "-" * 10)
print(img_binary.dtype)
print(img_binary)
#print(img_binary.dtype)
# cv2.imshow('img binary',img_binary)
# cv2.waitKey(0)


# 通过numpy转换为二值图
img_binary_rgb = np.where(img_gray_rgb >= 0.5, 1, 0)
img_binary_rgb = img_binary_rgb.astype(np.float64)
print("-" * 10 + "img binary rgb" + "-" * 10)
print(img_binary_rgb.dtype)
print(img_binary_rgb)
# cv2.imshow('img binary rgb',img_binary_rgb)
# cv2.waitKey(0)


plt.subplot(251)
plt.imshow(image_rgb)
plt.subplot(252)
plt.imshow(image_gray_rgb, cmap = 'gray')
plt.subplot(253)
plt.imshow(img_gray_rgb, cmap = 'gray')
plt.subplot(254)
plt.imshow(image_binary_rgb, cmap = 'gray')
plt.subplot(255)
plt.imshow(img_binary_rgb, cmap = 'gray')
plt.subplot(256)
plt.imshow(image)
plt.subplot(257)
plt.imshow(img_gray, cmap = 'gray')
plt.subplot(258)
plt.imshow(image_binary, cmap = 'gray')
plt.subplot(259)
plt.imshow(img_binary, cmap = 'gray')
plt.show()
