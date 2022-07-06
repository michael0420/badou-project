from sklearn.decomposition import PCA
import numpy as np
x = np.array([[15,5,7],
             [13,4,6],
             [14,7,8],
             [13,6,4],
             [11,6,5]])
k = 2
pca = PCA(n_components=2)
pca.fit(x)
print(pca.explained_variance_ratio_)
print(pca.fit_transform(x))
