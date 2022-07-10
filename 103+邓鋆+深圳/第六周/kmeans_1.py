import cv2
import numpy as np
import matplotlib.pyplot as plt


img  = cv2.imread("6.png",0)

h = int(img.shape[0] * 0.8)
w = int(img.shape[1] * 0.8)
img = cv2.resize(img,(w,h))

#cv2.imshow('src',img)
#cv2.waitKey(0)

#设置中心点个数
k = 3

#图像二维像素转换为一维
data = img.reshape((h * w, 1))
data = np.float32(data)


#停止条件 (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

#设置标签
flags = cv2.KMEANS_RANDOM_CENTERS


#K-Means聚类 聚集成4类
compactness, labels, centers = cv2.kmeans(data, k, None, criteria, 10, flags)

#生成最终图像
dst = labels.reshape((img.shape[0], img.shape[1]))

plt.imshow(dst, 'gray')
plt.show()
#cv2.imshow('dst',dst)
#cv2.waitKey(0)