from sklearn.decomposition import PCA
import numpy as np

X = np.array([[-1, 2, 66, -1], [-2, 6, 58, -1], [-3, 8, 45, -2],
              [1, 9, 36, 1], [2, 10, 62, 1], [3, 5, 83, 2]])  # 导入数据，维度为4
pca = PCA(n_components=3)  # 降到2维
pca.fit(X)
X_new = pca.fit_transform(X)
print(X_new)
