#!/usr/bin/env python
# -*-coding:utf-8-*-

from sklearn.cluster import KMeans
"""
第一部分：数据集
X表示二维矩阵数据，篮球运动员比赛数据
总共20行，每行两列数据
第一列表示球员每分钟助攻数：assists_per_minute
第二列表示球员每分钟得分数：points_per_minute
"""

'''KMeans聚类'''
X = [[0.0888, 0.5885],
     [0.1399, 0.8291],
     [0.0747, 0.4974],
     [0.0983, 0.5772],
     [0.1276, 0.5703],
     [0.1671, 0.5835],
     [0.1306, 0.5276],
     [0.1061, 0.5523],
     [0.2446, 0.4007],
     [0.1670, 0.4770],
     [0.2485, 0.4313],
     [0.1227, 0.4909],
     [0.1240, 0.5668],
     [0.1461, 0.5113],
     [0.2315, 0.3788],
     [0.0494, 0.5590],
     [0.1107, 0.4799],
     [0.1121, 0.5735],
     [0.1007, 0.6318],
     [0.2567, 0.4326],
     [0.1956, 0.4280]]
clf = KMeans(n_clusters=3, max_iter=100)  # 类簇为3，聚成3类。算法每次迭代的最大次数100
y_pred = clf.fit_predict(X)  # 聚类
print("y_pred=", y_pred)

'''绘图'''
import numpy as np
import matplotlib.pyplot as plt

x = [n[0] for n in X]  # 获取数据集X的第一列
print(x)
y = [n[1] for n in X]  # 获取数据集X的第二列
print(y)

plt.scatter(x, y, c=y_pred, marker="*")      #绘制散点图
plt.title("Kmeans-Basketball Data")     #设置标题
plt.xlabel("assists_per_minute")   #设置x轴
plt.ylabel("points_per_minute")    #设置y轴
plt.legend(['A','B','C'])     #设置图例
plt.show()