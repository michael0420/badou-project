#!/usr/bin/python
# encoding=gbk

import matplotlib.pyplot as plt
import sklearn.decomposition as dp
from sklearn.datasets import load_iris

'''鸢尾花数据集降维'''
x, y = load_iris(return_X_y=True)  # 加载数据集，x表示数据集中的属性数据，y表示数据标签
# iris=load_iris()
# X=iris.data
# Y=iris.target
pca = dp.PCA(n_components=2)  # 加载pca算法，设置降维后主成分数目为2
reduced_x = pca.fit_transform(x)  # 对数据集进行降维，保存在reduced_x中
red_x, red_y = [], []
green_x, green_y = [], []
blue_x, blue_y = [], []
for i in range(len(reduced_x)):  # 将降维后的数据点保存在不同的表中
    if y[i] == 0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    elif y[i] == 1:
        green_x.append(reduced_x[i, 0])
        green_y.append(reduced_x[i, 1])
    else:
        blue_x.append(reduced_x[i, 0])
        blue_y.append(reduced_x[i, 1])
plt.scatter(red_x, red_y, s=50, c="r", marker="+")
plt.scatter(green_x, green_y, s=50, c="g", marker="D")
plt.scatter(blue_x, blue_y, s=50, c="b", marker=".")
plt.show()
print("降维后的样本集：\n", reduced_x)
