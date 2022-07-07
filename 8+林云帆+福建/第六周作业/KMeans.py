import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('lenna.png', 0)
rows,cols = img.shape[:]
data = img.reshape((rows * cols, 1))
data = np.float32(data)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
flag = cv2.KMEANS_RANDOM_CENTERS
compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flag)
dst=labels.reshape((img.shape[0],img.shape[1]))
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
titles = [u'原始图像',u'聚类图像']
image = [img, dst]
for i in range(2):
    plt.subplot(1,2,i+1),plt.imshow(image[i],'gray'),
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()