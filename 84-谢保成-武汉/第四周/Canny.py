import cv2
import numpy as np
import math


# ==================边缘检测(canny)========================
'''
1.  边缘检测:
    需要灰度图，将其高斯滤波化，实现canny
'''
# 高斯滤波化
filePath = '../image/lenna.png'
img_color = cv2.imread(filePath, 1)
imgInfo = img_color.shape
height = imgInfo[0]
width = imgInfo[1]
gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
imgG = cv2.GaussianBlur(gray, (3, 3), 0)
res = cv2.Canny(img_color, 50, 50)
cv2.imshow('margin check(canny) ', res)

# ==================边缘检测(算法原理)========================
# sobel算子模板 分为竖直方向与水平方向
'''
[            |    [
 1  2  1     |      1  0  -1
 0  0  0     |      2  0  -2
-1 -2 -1     |      1  0  -1
]            |              ]
'''
# 图片卷积: 当前像素乘以模板再求和
'''
[1 2 3 4] * [a b c b] = 1*a + 2*b + 3*c + 4*b = dst 
'''
# 阈值判决
'''
sqrt(a*a(竖直方向上卷积的结果) + b*b(水平方向上卷积的结果)) = f(幅值) > th(判决阈值) -> 表示为边缘
'''
gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
dst = np.zeros((height, width, 1), np.uint8)
for i in range(height - 2):
    for j in range(width - 2):
        # 竖直方向的卷积
        gy = gray[i, j] * 1 + gray[i, j + 1] * 2 + gray[i, j + 2] * 1 - gray[i + 2, j] * 1 - gray[i + 2, j + 1] * 2 - \
             gray[i + 2, j + 2] * 1
        # 水平方向的卷积
        gx = gray[i, j] + gray[i + 1, j] * 2 + gray[i + 2, j] - gray[i, j + 2] - gray[i + 1, j] * 2 - gray[i, j]
        # 计算梯度
        grad = math.sqrt(gy * gy + gx * gx)
        # 做阈值判决
        if grad > 20:
            dst[i, j] = 255
        else:
            dst[i, j] = 0
cv2.imshow('margin check(theory)', dst)