import cv2
import numpy as  np

img = cv2.imread('photo1.jpg')
print(img.shape)
img1 = np.copy(img) # 或者 img1 = img.copy() 都可以
# src 和 dst 为图像定点的四个坐标

src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

m = cv2.getPerspectiveTransform(src, dst) # 生成透视矩阵
transform_img = cv2.warpPerspective(img1, m, (337, 488))
cv2.imshow('src', img1)
cv2.imshow('dst', transform_img)
cv2.waitKey(0)