#!/usr/bin/python
# -*-encoding:utf-8-*-

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

'''
linkage(y, method=’single’, metric=’euclidean’) 共包含3个参数: 
1. y是距离矩阵,可以是1维压缩向量（距离向量），也可以是2维观测向量（坐标矩阵）。
若y是1维压缩向量，则y必须是n个初始观测值的组合，n是坐标矩阵中成对的观测值。
2. method是指计算类间距离的方法。
'''
'''
fcluster(Z, t, criterion=’inconsistent’, depth=2, R=None, monocrit=None) 
1.第一个参数Z是linkage得到的矩阵,记录了层次聚类的层次信息; 
2.t是一个聚类的阈值-“The threshold to apply when forming flat clusters”。
'''

'''层次聚类'''
X = [[1, 2], [3, 2], [4, 4], [1, 2], [1, 3]]
Z = linkage(X, method="ward")  # 进行层次聚类
f = fcluster(Z, 2, criterion="distance")  # 根据Z得到聚类结果
plt.figure(figsize=(5, 3))  # 新建一个图像,并设置宽和高
dn=dendrogram(Z)   #将层次聚类绘制为树状图
print("Z:\n",Z)
print("f:\n",f)
plt.show()
