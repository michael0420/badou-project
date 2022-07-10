#!/usr/bin/env python
# encoding=utf-8

import numpy as np

'''
使用PCA求样本矩阵X的K阶降维矩阵Z
'''


class CPCA(object):
    '''
    Note:请保证输入的样本矩阵X shape=(m, n)，m行样例，n个特征
    '''

    def __init__(self, X, K):
        self.X = X  # 样本矩阵X
        self.K = K  # K阶降维矩阵的K值
        self.centerX = []  # 样本矩阵X中心化
        self.C = []  # 样本矩阵X的协方差矩阵C
        self.U = []  # 样本矩阵X的特征向量矩阵U
        self.Z = []  # 样本矩阵X的K阶降维矩阵Z

        self.centerX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()

    def _centralized(self):
        '''矩阵X的中心化'''
        print("样本矩阵X:\n", self.X)
        centerX = []
        mean = np.array(np.mean(self.X, 0))  # 求每一列的平均值
        # mean = np.array([np.mean(attr) for attr in self.X.T])
        print("样本集的特征均值:\n", mean)
        centerX = self.X - mean  # 样本集中心化
        print("样本矩阵X的中心化centrX:\n", centerX)
        return centerX

    def _cov(self):
        '''求样本矩阵X的协方差矩阵C'''
        # 样本集样例总数
        ns = np.shape(self.centerX)[0]
        # 计算样本矩阵X的协方差矩阵C
        C = np.dot(self.centerX.T, self.centerX) / (ns - 1)
        print("样本集协方差矩阵C:\n", C)
        return C

    def _U(self):
        '''样本矩阵X的特征向量矩阵U'''
        # 计算协方差矩阵C的特征值和特征向量
        w, v = np.linalg.eig(self.C)
        print("协方差矩阵C的特征值：\n", w)
        print("协方差矩阵C的特征向量：\n", v)
        # 特征值降序排列索引
        ind = np.argsort(-1 * w)
        # 构建K阶降维转换矩阵
        UT = [v[:, ind[i]] for i in range(self.K)]
        # 转置，作为列向量构建特征向量矩阵U
        U = np.transpose(UT)
        # U = UT.transpose()
        print("样本矩阵X的%s阶特征向量矩阵U：\n" % self.K, U)
        return U

    def _Z(self):
        '''样本集的的K阶降维矩阵Z'''
        Z = np.dot(self.centerX, self.U)
        print("X shape:", np.shape(self.X))
        print("U shape:", np.shape(self.U))
        print("Z shape:", np.shape(Z))
        print("样本集的的%s阶降维矩阵Z:\n" % self.K, Z)
        return Z


if __name__ == "__main__":
    '''10样本3特征的样本集, 行为样例，列为特征维度'''
X = np.array([[10, 15, 29],
              [15, 46, 13],
              [23, 21, 30],
              [11, 9, 35],
              [42, 45, 11],
              [9, 48, 5],
              [11, 21, 14],
              [8, 5, 15],
              [11, 12, 21],
              [21, 20, 25]])
K = np.shape(X)[1] - 1
print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
pca = CPCA(X, K)
