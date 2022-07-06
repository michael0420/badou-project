import cv2
import numpy as np

# 目标降到的特征纬度数
k = 2
# 样本
sample = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2],
    [4.7, 3.2, 1.3, 0.2],
    [4.6, 3.1, 1.5, 0.2],
    [5.0, 3.6, 1.4, 0.2],
    [5.4, 3.9, 1.7, 0.4],
    [4.6, 3.4, 1.4, 0.3],
    [5.0, 3.4, 1.5, 0.2],
    [4.4, 2.9, 1.4, 0.2],
    [4.9, 3.1, 1.5, 0.1],
    [5.4, 3.7, 1.5, 0.2],
    [4.8, 3.4, 1.6, 0.2],
    [4.8, 3.0, 1.4, 0.1],
    [4.3, 3.0, 1.1, 0.1],
    [5.8, 4.0, 1.2, 0.2],
])

# 中心化
print("中心化", "-" * 20)
center = sample - sample.mean(axis=0)
print(center)

# 求协方差矩阵 D = ZT*Z/m
print("求协方差矩阵", "-" * 20)
m = center.shape[0]
cov = np.dot(center.T, center) / m
print(cov)

# 计算特征值和特征向量
print("计算特征值和特征向量", "-" * 20)
feature_values, feature_vectors = np.linalg.eig(cov)
idx = np.argsort(feature_values)[::-1]
# 降维矩阵
reducer = np.transpose([feature_vectors[:, idx[i]] for i in range(k)])
print(reducer)

# 降维后得到最终目标样本
print("降维后得到最终目标样本", "-" * 20)
sample_result = np.dot(sample, reducer)
print(sample_result)
