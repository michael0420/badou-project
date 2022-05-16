from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

#原图
plt.subplot(221)
img = plt.imread("jjy.jpg")
# img = cv2.imread("lenna.png", False)
plt.imshow(img)
print("---image lenna----")
print(img)

# 灰度化
img = cv2.imread("jjy.jpg")
img_gray = rgb2gray(img)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_gray = img
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("---image gray----")
print(img_gray)

#二值化
img_binary = np.where(img_gray >= 0.5, 1, 0)
print("-----imge_binary------")
print(img_binary)
print(img_binary.shape)
plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.savefig(r"F:\anaconda3\自学脚本集\badouai\cv\JJY_TEST.JPG")
plt.show()








