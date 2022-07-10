import numpy as np
from sklearn.decomposition import PCA

X = np.array([[10, 15, 20, 16],
              [25, 29, 15, 11],
              [22, 23, 24, 2],
              [11, 19, 5, 23],
              [23, 21, 16, 15]
              ])
pca = PCA(n_components=2)
pca.fit(X)
newX=pca.fit_transform(X)   #降维后的数据
print(pca.explained_variance_ratio_)  #输出贡献率
print(newX)                  #输出降维后的数据