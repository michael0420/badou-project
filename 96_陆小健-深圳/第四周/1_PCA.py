# PCA手写过程以及利用sklearn库，看两种方法对同一数据的计算结果是否一样

import numpy as np
from sklearn.decomposition import PCA

class CPCA(object):
    def __init__(self, X, K):
        self.X = X       #样本矩阵X
        self.K = K       #K阶降维矩阵的K值
        self.centrX = [] #矩阵X的中心化
        self.C = []      #样本集的协方差矩阵C
        self.U = []      #样本矩阵X的降维转换矩阵
        self.Z = []      #样本矩阵X的降维矩阵Z
        
        self.centrX = self._centralized()
        self.C = self._cov()
        self.U = self._U()
        self.Z = self._Z() #Z=XU求得
        
    def _centralized(self):
        '''矩阵X的中心化'''
        centrX = []
        mean = np.array([np.mean(attr) for attr in self.X.T]) #样本集的特征均值
        print('样本集的特征均值:\n',mean)
        centrX = self.X - mean ##样本集的中心化
        # print('样本矩阵X的中心化centrX:\n', centrX)
        return centrX
        
    def _cov(self):
        '''求样本矩阵X的协方差矩阵C'''
        #样本集的样例总数
        ns = np.shape(self.centrX)[0]
        #样本矩阵的协方差矩阵C
        C = np.dot(self.centrX.T, self.centrX)/(ns - 1)
        # print('样本矩阵X的协方差矩阵C:\n', C)
        return C
        
    def _U(self):
        '''求X的降维转换矩阵U, shape=(n,k), n是X的特征维度总数，k是降维矩阵的特征维度'''
        #先求X的协方差矩阵C的特征值和特征向量
        a,b = np.linalg.eig(self.C) #特征值赋值给a，对应特征向量赋值给b。函数doc：https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linalg.eig.html 
        print('样本集的协方差矩阵C的特征值:\n', a)
        # print('样本集的协方差矩阵C的特征向量:\n', b)
        #给出特征值降序的topK的索引序列
        ind = np.argsort(-1*a)
        #构建K阶降维的降维转换矩阵U
        UT = [b[:,ind[i]] for i in range(self.K)]
        U = np.transpose(UT)
        return U
        
    def _Z(self):
        '''按照Z=XU求降维矩阵Z, shape=(m,k), n是样本总数，k是降维矩阵中特征维度总数'''
        Z = np.dot(self.X, self.U)
        print('样本矩阵X的降维矩阵Z:\n', Z)
        return Z
X = np.array([[-1,2,66,-1], 
              [-2,6,58,-1], 
              [-3,8,45,-2], 
              [1,9,36,1], 
              [2,10,62,1], 
              [3,5,83,2]])  #导入数据，维度为4
K = np.shape(X)[1] - 1
pca = CPCA(X,K)


# 使用sklearn库函数的PCA函数
pca_1 = PCA(n_components=np.shape(X)[1]-1)   #降到2维
pca_1.fit(X)                  #训练
newX=pca_1.fit_transform(X)   #降维后的数据
# PCA(copy=True, n_components=2, whiten=False)
print('特征值大小=',pca_1.explained_variance_ratio_)  #输出贡献率，就是特征值大小
print('使用sklearn库函数的降维矩阵newX:\n',newX)                  #输出降维后的数据

