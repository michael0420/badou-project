import numpy as np
from sklearn.decomposition import PCA


class CPCA(object):
    def __init__(self, x, k):
        self.x = x
        self.k = k
        self.centre = []
        self.c = []
        self.u = []
        self.z = []

        self.centre = self._centralized()
        self.c = self._conv()
        self.u = self._u()
        self.z = self._z()

    def _centralized(self):
        mean = np.array([np.mean(attr) for attr in self.x.T])
        centralized_x = self.x - mean
        return centralized_x

    def _conv(self):
        ns = np.shape(self.centre)[0]
        c = np.dot(self.centre.T, self.centre)/(ns-1)
        return c

    def _u(self):
        a, b = np.linalg.eig(self.c)
        ind = np.argsort(-1*a)
        ut = [b[:, ind[i]] for i in range(self.k)]
        u = np.transpose(ut)
        return u

    def _z(self):
        z = np.dot(self.x, self.u)
        print('x shape:', np.shape(self.x))
        print('u shape:', np.shape(self.u))
        print('z shape:', np.shape(z))
        print('样本矩阵x降维矩阵z：', z)
        return z


if __name__ == '__main__':
    x = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9,  35],
                  [42, 45, 11],
                  [9,  48, 5],
                  [11, 21, 14],
                  [8,  5,  15],
                  [11, 12, 21],
                  [21, 20, 25]])
    k = np.shape(x)[1] - 1
    print('样本集(10行3列，10个样例，每个样例3个特征):\n', x)
    pca = CPCA(x, k)
