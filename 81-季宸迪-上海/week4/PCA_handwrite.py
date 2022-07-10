# PCA handwrite

import numpy as np

class PCA:
    def __init__(self, X, K):
        self.X = X # sample matrix
        self.K = K # dimension target
        self.centraX = [] # centerlized sample matrix
        self.C = [] # covariance matrix
        self.W = [] # tranfer matrix
        self.Z = [] # sample matrix after dimensionality reduction

        centraX = self._centralization(X)
        C = self._cov(centraX)
        W = self._W(C, K)
        Z = self._Z(centraX, W)

    # calculate centerlized sample matrix
    def _centralization(self, X):
        mean = np.array([np.mean(feat) for feat in X.T])
        centraX = X - mean
        print('样本矩阵X的中心化centrX:\n', centraX)
        return centraX

    # calculate covariance matrix
    def _cov(self, centraX):
        C = np.dot(centraX.T, centraX) / (len(centraX)-1)
        print('样本矩阵X的协方差矩阵C:\n', C)
        return C

    # calculate the tranfer matrix
    def _W(self, C, K):
        a, b = np.linalg.eig(C)
        index = np.argsort(-1*a)
        WT = [b[:,index[i]] for i in range(K)]
        W = np.transpose(WT)
        print('%d阶降维转换矩阵U:\n'%self.K, W)
        return W

    # calculate the matrix after PCA
    def _Z(self, centraX, W):
        Z = np.dot(centraX, W)
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z

if __name__=='__main__':
    '10样本3特征的样本集, 行为样例，列为特征维度'
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9,  35],
                  [42, 45, 11],
                  [9,  48, 5],
                  [11, 21, 14],
                  [8,  5,  15],
                  [11, 12, 21],
                  [21, 20, 25]])
    K = np.shape(X)[1] - 1
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', X)
    pca = PCA(X,K)