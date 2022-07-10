#!/usr/bin/python
# encodig=utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
在OpenCV中，Kmeans()函数原型如下所示：
retval, bestLabels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
    data表示聚类数据，最好是np.flloat32类型的N维点集
    K表示聚类类簇数
    bestLabels表示输出的整数数组，用于存储每个样本的聚类标签索引
    criteria表示迭代停止的模式选择，这是一个含有三个元素的元组型数。格式为（type, max_iter, epsilon）
        其中，type有如下模式：
         —–cv2.TERM_CRITERIA_EPS :精确度（误差）满足epsilon停止。
         —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
         —-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
    attempts表示重复试验kmeans算法的次数，算法返回产生的最佳结果的标签
    flags表示初始中心的选择，两种方法是cv2.KMEANS_PP_CENTERS ;和cv2.KMEANS_RANDOM_CENTERS
    centers表示集群中心的输出矩阵，每个集群中心为一行数据
'''

'''opencv K-Means聚类'''
img = cv2.imread("lenna.png", 0)    #读取原始图像灰度颜色
rows, cols = img.shape[:]   #获取图像高度、宽度
data = img.reshape([rows * cols, 1])
data = np.float32(data)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)  # 设置迭代停止条件
flags = cv2.KMEANS_RANDOM_CENTERS  # 设置每次迭代随机选择初始中心
compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)  # K-Means聚类成4类
img_dst = labels.reshape((rows, cols))  # 生成最终图像
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置正常显示中文标签
image = [img, img_dst]
titles = [u"原始图像", u"聚类图像"]
# titles = ["原始图像", "聚类图像"]
for i in range(2):
    plt.subplot(1, 2, i + 1), plt.imshow(image[i], "gray")  # 显示图像
    plt.title(titles[i])  # 设置标题
    plt.xticks([]), plt.yticks([])  # 设置x，y轴刻度
plt.show()
