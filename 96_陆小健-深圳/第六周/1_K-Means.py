# coding: utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt

#读取原始图像灰度颜色
img = cv2.imread('lenna.png', 1) 
img_gray = cv2.imread('lenna.png', 0) 

#图像二维像素转换为一维
data = img.reshape((-1,3))
data = np.float32(data)
data_gray = img_gray.reshape((img_gray.shape[0] * img_gray.shape[1], 1))
data_gray = np.float32(data_gray)

#停止条件 (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#设置标签
flags = cv2.KMEANS_RANDOM_CENTERS

#K-Means聚类 将BGR图像变成4/32分类
compactness_1, labels_1, centers_1 = cv2.kmeans(data, 4, None, criteria, 10, flags)
compactness_2, labels_2, centers_2 = cv2.kmeans(data, 32, None, criteria, 10, flags)
#K-Means聚类 将灰度图图像变成4/32分类
compactness_3, labels_3, centers_3 = cv2.kmeans(data, 4, None, criteria, 10, flags)
compactness_4, labels_4, centers_4 = cv2.kmeans(data, 32, None, criteria, 10, flags)

#生成最终图像 
centers_1= np.uint8(centers_1)  #向下取整数
res_1 = centers_1[labels_1.flatten()]  #np.flatten()，该函数返回一个折叠成 一维 的数组
centers_2 = np.uint8(centers_2)  #向下取整数
res_2 = centers_2[labels_2.flatten()]
dst_1 = res_1.reshape((img.shape))
dst_2 = res_2.reshape((img.shape))
dst_3 = labels_3.reshape((img_gray.shape))
dst_4 = labels_4.reshape((img_gray.shape))

#图像转换为RGB显示
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst_1 = cv2.cvtColor(dst_1, cv2.COLOR_BGR2RGB)
dst_2 = cv2.cvtColor(dst_2, cv2.COLOR_BGR2RGB)

#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

#显示图像
titles = [u'原始图像', u'聚类图像 K=4', u'聚类图像 K=32', u'原始灰度图像', u'聚类图像 K=4', u'聚类图像 K=32']  
images = [img, dst_1, dst_2, img_gray, dst_3, dst_4]  
for i in range(3):  
   plt.subplot(2,3,i+1), plt.imshow(images[i]), 
   plt.title(titles[i])  
   plt.xticks([]),plt.yticks([])  
for i in range(3):  
   plt.subplot(2,3,i+4), plt.imshow(images[i+3], 'gray'), 
   plt.title(titles[i+3])  
   plt.xticks([]),plt.yticks([])  
plt.savefig('result.jpg')
plt.show()
