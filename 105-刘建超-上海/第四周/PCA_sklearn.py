#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
from sklearn.decomposition import PCA

X = np.array([[10, 15, 29], [15, 46, 13], [23, 21, 30], [11, 9, 35], [42, 45, 11], [9, 48, 5], [11, 21, 14], [8, 5, 15],
              [11, 12, 21], [21, 20, 25]])
pca = PCA(n_components=2)  # 降到2维
pca.fit(X)  # 训练数据
newX=pca.fit_transform(X)   #对数据降维
print("PCA_sklearn特征值：\n",pca.explained_variance_)
print("PCA_sklearn特征向量：\n",pca.components_)
print("各成分影响占比：\n",pca.explained_variance_ratio_)
print("PCA_sklearn降维后的样本集：\n",newX)