import numpy as np
import sklearn.decomposition as dp


def pca_api(x,k):
    pca = dp.PCA(n_components=k)
    pca.fit(x)  # 训练
    newx = pca.fit_transform(x)  # 新数据
    # print(pca.explained_variance_ratio_)
    return newx


def pca_detail(x,k):
    x = x-x.mean(axis=0)
    var = np.dot(x.T,x)/x.shape[0]
    eig,eig_vect = np.linalg.eig(var)
    idx = np.argsort(-eig) #返回升序索引，所以加负号
    sav_vect = eig_vect[:,idx[:k]]
    newx = np.dot(x,sav_vect)
    return newx


# x = np.array([[1,2,3],[1,3,4],[3,1,4],[1,4,6]])  # 4*3 ->4*2

x = np.array([[10,15,29],[15,46,13],[23,21,30],[11,9,35],[42,45,11],[9,48,5],[11,21,14],[8,5,15],[11,12,21],[21,20,25]])


newx1 = pca_api(x,2)
newx2 = pca_detail(x,2)

print(newx1)
print(newx2)
#结果完全一样，输入数据不同时会有不一样的结果



