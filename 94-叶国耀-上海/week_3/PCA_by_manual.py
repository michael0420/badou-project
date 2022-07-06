import numpy as np
import sklearn.decomposition as dp
# 输入矩阵
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
print("输入大小", X.shape)
# 降维维度
K = 2
# 中心化矩阵
mean = np.array([np.mean(X_dim) for X_dim in X.T])
X_zero = X - mean
# 求协方差矩阵
C = np.dot(X_zero.T, X_zero)/(X_zero.shape[1]-1)
# 求协方差矩阵的特征值和特征向量 |C-aE|=0
a,b = np.linalg.eig(C)  # a为特征值，b为特征向量
# 对特征值从大到小进行排序
ind = np.argsort(-1*a)
# 取前K个特征向量组成新的特征向量
b_new = np.array([b[:,ind[k]] for k in range(K)])
# 通过新的特征向量求降维后的样本
Y = np.dot(X_zero, b_new.T)

# sklearn
pca=dp.PCA(n_components=K) #加载pca算法，设置降维后主成分数目为2
Y_sk =pca.fit_transform(X)