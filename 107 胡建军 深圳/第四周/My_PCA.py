import numpy as np

class My_PCA(object):
    def __init__(self, X, K):
        self.X = X  #样本矩阵X
        self.K = K  #K阶降维矩阵的K值，K个主成分
        self.centrX = []    #矩阵X的中心化
        self.C = [] #样本集的协方差矩阵C
        self.U = [] #样本矩阵X的降维转换矩阵
        self.Z = [] #样本矩阵X的降维矩阵Z
        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()  # Z=XU求得

    def _centralized(self):
        '''
        样本矩阵均值归一化
        :return:
        '''
        centrX = self.X - np.mean(self.X, axis=0)
        return centrX

    def _cov(self):
        '''
        样本协方差矩阵
        :return:
        '''
        n = np.shape(self.centrX)[0]
        C = np.dot(self.centrX.T, self.centrX)/(n-1)
        return C

    def _U(self):
        '''
        求解特征空间
        :return:
        '''
        a, v = np.linalg.eig(self.C)    #对协方差矩阵进行特征分解
        U = v[:, :self.K]  #选择前K个特征向量组成特征空间
        return U

    def _Z(self):
        '''
        按照Z=XU求降维矩阵Z, shape=(m,k), n是样本总数，k是降维矩阵中特征维度总数
        :return:
        '''
        Z = np.dot(self.X, self.U)
        return Z

if __name__ == '__main__':
    '10样本3特征的样本集, 行为样例，列为特征维度'
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
    pca = My_PCA(X, K)
    print('-----特征空间-------\n', pca.U)
    print('-----降维后数据------\n', pca.Z)



