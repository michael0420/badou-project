import cv2
import numpy as np

# 灰度化
img = cv2.imread("lenna.png")
height = img.shape[0]
width = img.shape[1]
img_g = np.zeros([height, width], img.dtype)  # 创建一张和当前图片大小一样的单通道图片
print(height, width)
for i in range(height):
    for j in range(width):
        m = img[i, j]  # 取出当前high和wide中的BGR坐标
        img_g[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)  # 将BGR坐标转化为gray坐标并赋值给新图像
# # cv2.imwrite("a.jpg", img_g)

# 二值化
rows, cols = img_g.shape
img_b = np.zeros([rows,cols],img_g.dtype)
for i in range(rows):
    for j in range(cols):
        if (img_g[i, j] <= 0.5):
            img_b[i, j] = 0
        else:
            img_b[i, j] = 1
cv2.imwrite("b.jpg", img_b)