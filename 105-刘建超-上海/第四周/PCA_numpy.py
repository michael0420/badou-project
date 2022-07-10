#!/usr/bin/python
# coding=utf-8

import numpy as np

'''
使用PCA求样本矩阵X的K阶降维矩阵Z
'''


class PCA():
    def __init__(self, n_components):
        self.n_components = n_components

    def fit_transform(self, X):
        '''矩阵X的中心化'''
        self.centerX = X - X.mean(axis=0)
        '''求协方差矩阵'''
        self.covariance = np.dot(self.centerX.T, self.centerX) / (X.shape[0] - 1)
        '''求协方差矩阵的特征值、特征向量'''
        w, v = np.linalg.eig(self.covariance)
        '''特征值降序排列索引'''
        ind = np.argsort(-w)
        '''构建特征向量矩阵'''
        self.components_ = v[:, ind[:self.n_components]]
        '''求样本集的降维矩阵'''
        return np.dot(self.centerX, self.components_)


# 调用
pca = PCA(n_components=2)
X = np.array([[10, 15, 29], [15, 46, 13], [23, 21, 30], [11, 9, 35], [42, 45, 11], [9, 48, 5], [11, 21, 14], [8, 5, 15],
              [11, 12, 21], [21, 20, 25]])
Z = pca.fit_transform(X)
print("样本集X的降维矩阵：\n",Z)

