

import numpy as np

'''
使用PCA求样本矩阵X的K阶降维矩阵Z
'''


class CPCA(object):
    '''
  输入的样本矩阵X shape=（m，n）m为行样例，n为特征
  也可以输入图像矩阵进行降维处理
  '''

    def __init__(self, X, K):
        '''
      :param X:样本矩阵X
      :param K:将样本矩阵X降维K列
      '''
        self.X = X  # 样本矩阵X
        self.K = K  # PCA之后降维K值
        self.centreX = []  # 中心化之后的样本矩阵
        self.C = []  # 协方差矩阵C
        self.U = []  # 样本矩阵X的降维转换矩阵U
        self.Z = []  # 样本矩阵X降维之后的矩阵Z

        self.centreX = self._centeralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z()  # Z=XU求得

    def _centeralized(self):
        '''样本矩阵X中心化'''
        print('样本矩阵X：\n', X)
        mean = np.array([np.mean(attr)for attr in self.X.T] )  # 样本集的特征均值
        print('样本矩阵的特征均值：\n', mean)
        centreX = self.X - mean  # 样本集中心化
        print('中心化后的样本矩阵centreX：\n', centreX)
        return centreX
    def _cov(self):
        '''求样本矩阵X的协方差矩阵'''
        #样本集合的样例总数
        ns = np.shape(self.centreX)[0]
        C = np.dot(self.centreX.T,self.centreX)/(ns-1)
        print('样本集合X的协方差矩阵：\n',C)
        return C
    def _U(self):
        '''求X得降维转换矩阵U，shape=（n，k），n是X的特征维度总数，k是降维矩阵的特征维度'''
        #先求X的协方差矩阵的特征值和特征向量
        a,b = np.linalg.eig(self.C)  #a是协方差矩阵的特征值，b是协方差矩阵的特征向量
        print('特征值矩阵a：\n', a)
        print('特征向量矩阵b：\n', b)
        #给特征值降序到的TOPK的索引序列
        ind = np.argsort(-1*a)
        #构建K阶降维的降维转换矩阵U
        UT = [b[:,ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        print('%d阶降维转换矩阵U:\n' % self.K, U)
        return U
    def _Z(self):
        #Z = np.dot(self.X,self.U)
        Z = np.dot(self.centreX, self.U)
        print('X shape:',np.shape(self.X))
        print('U shape:',np.shape(self.U))
        print('Z shape:',np.shape(Z))
        print('样本矩阵X的降维矩阵Z：\n',Z)
        return Z

X = np.array([[10, 15, 20, 16],
              [25, 29, 15, 11],
              [22, 23, 24, 2],
              [11, 19, 5, 23],
              [23, 21, 16, 15]
              ])
K = np.shape(X)[1] - 2
print('原始样本集X为:\n', X)
pca = CPCA(X, K)
print('First->降维后的特征数据：\n', pca.Z)
reconstruct = np.dot(pca.Z, pca.U.T)
print('First->特征重构后的数据集\n', reconstruct)