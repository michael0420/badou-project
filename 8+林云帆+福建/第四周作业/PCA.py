import numpy as np


class PCA_detail():
    def __init__(self, X, K):
        self.X = X  # 样本矩阵X
        self.K = K  # 降维到K阶
        self.centr_X = []  # 样本矩阵X中心化
        self.C = []  # 样本的协方差矩阵C
        self.U = []  # 转换矩阵U
        self.Z = []  # 降维矩阵Z
        self.centr_X = self._centralized()
        self.C = self._C()
        self.U = self._U()
        self.Z = self._Z()

    def _centralized(self):
        centr_X = []
        mean = [np.mean(a) for a in self.X.T]
        centr_X = self.X - mean
        return centr_X

    def _C(self):
        ns = np.shape(self.centr_X)[0]
        C = np.dot(self.centr_X.T, self.centr_X) / (ns - 1)
        return C

    def _U(self):
        a, b = np.linalg.eig(self.C)
        index = np.argsort(-1 * a)
        UT = [b[:, index[j]] for j in range(self.K)]
        print(UT)
        U = np.transpose(UT)
        print(U.shape)
        return U

    def _Z(self):
        Z = np.dot(self.X, self.U)
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z


if __name__ == '__main__':
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
    pca = PCA_detail(X, K)