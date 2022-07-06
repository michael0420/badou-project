import numpy as np

def pca(X,K):
    centrX = centert(X)
    convX = cov(centrX)
    chartV = chart(convX,K)
    dataSet(X, chartV)

def centert(X):
    '''矩阵X的中心化'''
    mean = np.mean(X,axis=0)
    print("mean:" ,mean)
    centrX = X-mean
    print("样本中心化",centrX)  ##样本集的中心化
    return centrX

def cov(centrX):
    '''求样本矩阵X的协方差矩阵'''
    convX = np.cov(centrX.T)
    print("协方差矩阵",convX)
    return convX

def chart(convX,K):
    a, b = np.linalg.eig(convX)
    print('样本集的协方差矩阵的特征值:\n', a)
    print('样本集的协方差矩阵的特征向量:\n', b)

    ind = np.argsort(-1 * a)
    print("int",ind)

    UT = [b[:, ind[i]] for i in range(K)]

    chartX = np.transpose(UT)
    print('%d阶降维转换矩阵U:\n' % K, chartX)

    print("特征值评分",np.sum(a[:K])/np.sum(a))


    return chartX

def dataSet(X,chartV):

    Z = np.dot(X, chartV)
    print('样本矩阵X的降维矩阵Z:\n', Z)

X = np.array([[10, 15, 19, 48, 5],
              [15, 46, 13, 48, 5],
              [23, 21, 21, 9, 33],
              [11, 9, 35, 21, 5],
              [42, 45, 55, 15, 29],
              [9, 48, 5, 9, 35],
              [11, 21, 14, 98, 5],
              [8, 5, 15, 9, 35],
              [11, 12, 21, 78, 13],
              [21, 20, 25, 5, 11]])
              
              
              
            

'''X = np.array([[10, 15, 29],
              [15, 46, 13],
              [23, 21, 30],
              [11, 9, 35],
              [42, 45, 11],
              [9, 48, 5],
              [11, 21, 14],
              [8, 5, 15],
              [11, 12, 21],
              [21, 20, 25]])'''
pca(X,4)