# cluster.py
# 导入相应的包
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt

'''
linkage(y, method=’single’, metric=’euclidean’) 共包含3个参数: 
# 1. y是距离矩阵,可以是1维压缩向量（距离向量），也可以是2维观测向量（坐标矩阵）。
# 若y是1维压缩向量，则y必须是n个初始观测值的组合，n是坐标矩阵中成对的观测值。
# 2. method是指计算类间距离的方法。
1.第一个参数y为一个尺寸为(m,n)的二维矩阵。一共有n个样本，每个样本有m个维度。
2.参数method =
’single’：一范数距离
’complete’：无穷范数距离
’average’：平均距离
’centroid’：二范数距离
’ward’：离差平方和距离
3.返回值：(n-1)*4的矩阵Z
'''

'''
fcluster(Z, t, criterion=’inconsistent’, depth=2, R=None, monocrit=None) 
这个函数压平树状图
这种分配主要取决于距离阈值t——允许的最大簇间距离
1.参数Z是linkage函数的输出Z。
2.参数scalar：形成扁平簇的阈值。
3.参数criterion：
’inconsistent’：预设的，如果一个集群节点及其所有后代的不一致值小于或等于 t，那么它的所有叶子后代都属于同一个平面集群。当没有非单例集群满足此条件时，每个节点都被分配到自己的集群中。
’distance’：每个簇的距离不超过t
4.输出是每一个特征的类别。
'''

X = [[1, 2], [3, 2], [4, 4], [1, 2], [1, 3]]
Z = linkage(X, 'ward')  # Ward方差最小化算法
f = fcluster(Z, 4, 'distance')
# print(f)
print(Z)
fig = plt.figure(figsize=(5, 3))   # 自定义图像大小
dn = dendrogram(Z)   # 绘制层次聚类图
# print(dn)
plt.show()
