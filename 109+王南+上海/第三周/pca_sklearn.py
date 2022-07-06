import sklearn.decomposition as dp
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

sample = load_iris(return_X_y=False).data
pca = dp.PCA(n_components=2)
sample_pca = pca.fit_transform(sample)

print(sample_pca)

plt.scatter(sample_pca[:, 0], sample_pca[:, 1], c="r", marker=".")
plt.show()