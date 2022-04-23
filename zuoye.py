# -*- coding
"""
戎翀作业
彩色图像灰度化、二值化

"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

#原图
img = plt.imread("lenna.png")
plt.subplot(221)
plt.imshow(img)
print("---image lenna----")
print(img)

#灰度化
img_gray = rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("---image gray----")
print(img_gray)

#二值化
img_bi = np.where(img_gray >= 0.5, 1, 0)
plt.subplot(223)
plt.imshow(img_bi, cmap='gray')
plt.show()
print("---image binary----")
print(img_bi)

