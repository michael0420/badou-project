import numpy as np

class PCA():
    def __init__(self, X, K):

        self.X = X  # 样本矩阵X
        self.K = K  # K阶降维矩阵的K值
        self.centrXT = []  # 矩阵X的中心化
        self.C = []  # 样本集的协方差矩阵C
        self.U = []  # 样本矩阵X的降维转换矩阵
        self.Z = []  # 样本矩阵X的降维矩阵Z

        self.centrXT = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()  # Z=UX求得

    def _centralized(self):
        print('样本矩阵X:\n', self.X)
        centrXT = []
        mean = np.average(self.X, axis=0)
        print('样本集的特征均值:\n',mean)
        centrXT = np.transpose(self.X - mean)
        print('样本矩阵X的中心化centrXT:\n', centrXT)
        return centrXT

    def _cov(self):
        ns = np.shape(self.centrXT)[1]
        C = np.dot(self.centrXT, self.centrXT.T) / (ns - 1)     # ????????????!!!!!!!!!!!!
        print('样本矩阵X的协方差矩阵C:\n', C)
        return C

    def _U(self):
        a,b =np.linalg.eig(self.C)
        print('样本集的协方差矩阵C的特征值:\n', a)
        print('样本集的协方差矩阵C的特征向量:\n', b)
        ind = np.argsort(-1 * a)
        print("ind:",ind)
        U = [b[:,ind[i]] for i in range(self.K)]
        print('%d阶降维转换矩阵U:\n' % self.K, U)
        return U

    def _Z(self):
        Z = np.dot(self.U, self.centrXT)
        print('X shape:', np.shape(self.X))
        print('U shape:', np.shape(self.U))
        print('Z shape:', np.shape(Z))
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z


if __name__ == "__main__":
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
    print(X)
    K = X.shape[1] - 1
    pca = PCA(X, K)
