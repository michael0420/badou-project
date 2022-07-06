#!/usr/bin/env python
# coding=utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt

'''opencv K-Means聚类RGB'''
img = cv2.imread("lenna.png")   #读取原始图像
data = img.reshape((-1, 3))
data = np.float32(data)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)    # 设置迭代停止条件
flags = cv2.KMEANS_RANDOM_CENTERS   # 设置每次迭代随机选择初始中心
# K-Means聚类成2类
compactness2, labels2, centers2 = cv2.kmeans(data, 2, None, criteria, 10, flags)
# K-Means聚类成4类
compactness4, labels4, centers4 = cv2.kmeans(data, 4, None, criteria, 10, flags)
# K-Means聚类成8类
compactness8, labels8, centers8 = cv2.kmeans(data, 8, None, criteria, 10, flags)
# K-Means聚类成16类
compactness16, labels16, centers16 = cv2.kmeans(data, 16, None, criteria, 10, flags)
# K-Means聚类成64类
compactness64, labels64, centers64 = cv2.kmeans(data, 64, None, criteria, 10, flags)

# 浮点数转换回uint8
centers2 = np.uint8(centers2)
res = centers2[labels2.flatten()]
dst2 = res.reshape(img.shape)   # 生成最终图像

centers4 = np.uint8(centers4)
res = centers4[labels4.flatten()]
dst4 = res.reshape(img.shape)   # 生成最终图像

centers8 = np.uint8(centers8)
res = centers8[labels8.flatten()]
dst8 = res.reshape(img.shape)   # 生成最终图像

centers16 = np.uint8(centers16)
res = centers16[labels16.flatten()]
dst16 = res.reshape(img.shape)  # 生成最终图像

centers64 = np.uint8(centers64)
res = centers64[labels64.flatten()]
dst64 = res.reshape(img.shape)  # 生成最终图像

# 图像转换为RGB显示
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)
dst4 = cv2.cvtColor(dst4, cv2.COLOR_BGR2RGB)
dst8 = cv2.cvtColor(dst8, cv2.COLOR_BGR2RGB)
dst16 = cv2.cvtColor(dst16, cv2.COLOR_BGR2RGB)
dst64 = cv2.cvtColor(dst64, cv2.COLOR_BGR2RGB)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置正常显示中文标签
image = [img, dst2, dst4, dst8, dst16, dst64]
titles = [u"原始图像", u"聚类图像K=2", u"聚类图像K=4", u"聚类图像K=8", u"聚类图像K=16", u"聚类图像K=64"]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(image[i])   # 显示图像
    plt.title(titles[i])    # 设置标题
    plt.xticks([]),plt.yticks([])   # 设置x，y轴刻度
plt.show()
