import numpy as np

class PCA:
    def __init__(self,x,k):
        self.X = x - np.mean(x,axis=0)#中心化
        self.N = x.shape[0] #得到样本数量
        self.K = k #得到降维后样本特征维度
        self.Sigma = self.get_Sigama()
        self.Lambda,self.Alpha = self.get_Lambda_Alpha()
    #求协方差矩阵
    def get_Sigama(self):
        return np.dot(self.X.T,self.X)/(self.N-1)
    def show_Sigama(self):
        Sigama_ =  np.dot(self.X.T,self.X)/(self.N-1)
        print("协方差矩阵展示如下")
        print(Sigama_)
        print("协方差矩阵形状展示如下")
        print(Sigama_.shape)
    #求协方差矩阵的特征值和特征向量
    def get_Lambda_Alpha(self):
        Lambda_, Alpha_ = np.linalg.eig(self.Sigma) #求协方差矩阵的特征值和特征向量
        Lambda_argsorted = np.argsort(-1 * Lambda_) #将特征值降序排列
        Lambda_ = Lambda_[Lambda_argsorted]
        Alpha_ = Alpha_[:,Lambda_argsorted]
        return Lambda_,Alpha_
    def show_Lambda_Alpha(self):
        Lambda_,Alpha_ = np.linalg.eig(self.Sigma)
        Lambda_argsorted = np.argsort(-1 * Lambda_)
        Lambda_ = Lambda_[Lambda_argsorted]
        Alpha_ = Alpha_[:, Lambda_argsorted]
        print("特征值展示如下")
        print(Lambda_)
        print("特征向量展示如下")
        print(Alpha_)
        print("特征向量形状展示如下")
        print(Alpha_.shape)
    #求降维后的数据
    def get_new_x(self):
        Alpha_ = self.Alpha[:,0:self.K] #得到降维所用的特征向量
        return np.dot(self.X,Alpha_) #得到降维后的数据
    def show_new_x(self):
        Alpha_ = self.Alpha[:, 0:self.K]
        new_x = np.dot(self.X,Alpha_)
        print("降维所用的特征向量展示如下")
        print(Alpha_)
        print("降维所用的特征向量形状展示如下")
        print(Alpha_.shape)
        print("降维后的数据展示如下")
        print(new_x)
        print("降维后的数据形状展示如下")
        print(new_x.shape)
if __name__ == '__main__':

    x = np.array([[15,5,7],
                  [13,4,6],
                  [14,7,8],
                  [13,6,4],
                  [11,6,5]])
    k = 2
    pca  = PCA(x,k) #实例化定义的PCA类
    pca.show_Sigama() #展示协方差矩阵
    pca.show_Lambda_Alpha() #展示特征值和特征向量
    pca.show_new_x() #展示降维后的数据
    new_x = pca.get_new_x() #调用pca.get_new_x()函数得到降维后的数据
    print(new_x)


