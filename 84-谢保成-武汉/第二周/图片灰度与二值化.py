import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

filePath = '/84-谢保成-武汉/image/p1.png'
# ============================灰度化===========================================
'''
1.源码实现灰度化
'''
# 读取图片
img = cv2.imread(filePath)
# 获取图片的信息
imgInfo = img.shape
# 高度
height = int(imgInfo[0] * 0.5)
# 宽度
width = int(imgInfo[1] * 0.5)
# 将原图缩小一半
img = cv2.resize(img, (width, height))
# 创建一个和原图大小相同的矩阵，用于存放变化之后的灰度图
img_gray = np.zeros([height, width], img.dtype)
for i in range(0, height):
    for j in range(0, width):
        # 读取到原图的BGR像素点
        newImg = img[i, j]
        img_gray[i, j] = int(newImg[0] * 0.11 + newImg[1] * 0.59 + newImg[2] * 0.3)
# print('img_gray\n', img_gray)
cv2.imshow('img', img)
cv2.imshow('gray', img_gray)
cv2.waitKey(0)

'''
2.调用cv原有的api实现灰度化
'''
img_cv_api2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('cv_api2gray', img_cv_api2gray)
cv2.waitKey(0)

# ============================二值化===========================================
'''
利用cv实现二值化
'''
cv2.threshold(img_gray, 127, 255, 0, img_gray)
cv2.imshow('cv_img_binary', img_gray)
cv2.waitKey(0)

'''
源码实现二值化（先灰度化）
'''
# 实现灰度化
img_gray = rgb2gray(img)
rows, cols = img_gray.shape
for i in range(rows):
    for j in range(cols):
        if img_gray[i, j] <= 0.5:
            img_gray[i, j] = 0
        else:
            img_gray[i, j] = 1
plt.imshow(img_gray, cmap='gray')
plt.show()

'''
利用numpy实现二值化
'''
img_binary = np.where(img_gray >= 0.5, 1, 0)
plt.imshow(img_binary, cmap='gray')
plt.show()

'''
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
'''
