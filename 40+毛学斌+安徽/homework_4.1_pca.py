import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets._base import load_iris
import matplotlib.pyplot as plt


def pca_detail(data_in, k):
    """
    完整详细的 原始PCA 执行代码，主要用于特征降维，去除信息量小的特征，提高运算运算效率
    输入：
        data_in:需要特征降维的数据矩阵，此时特征维度较多，运算效率低
        k：希望降低后的特征维度，需要根据经验设定
    输出：
        降低维度后的数据矩阵，特征维度减小了，后面的预算速度会提升
    """
    row, column = data_in.shape
    print('0、处理前的数据的形状为：', data_in.shape)

    # 1、原始数据零均值化（中心化），即将均值调整为0，便于后期的运算，此时形状不变例如 m*n
    mean = np.mean(data_in, axis=0)  # 求平均值 axis=0，保留行，也就是每列求平均值
    data_center = data_in - mean
    print('1、零均值化后的数据形状为：', data_center.shape)

    # 2、利用所有的特征形成特征的协方差矩阵，此时为非对角矩阵 形状为n*n
    cov = np.dot(data_center.T, data_center) / (row - 1)  # 因为均值是零，且均值相乘包含求和的运算，协方差矩阵计算可以简化
    print('2、协方差矩阵的形状为：', cov.shape)

    # 3、求解协方差矩阵的特征值和特征向量,相当于映射到一个新的由特征向量组成的对角矩阵（特征向量彼此正交），形状为n*n
    eigenvalue, eigenvector = np.linalg.eig(cov)  # np.linalg.eig()函数求解后的特征值已经按照从大到小排列
    print('3.1 特征值为：', eigenvalue)
    print('3.2 特征向量为：\n', eigenvector)
    # 特征向量是列向量的组合,有时候可能有正负号问题，但是主要看分布，而非方向。
    # 对于压缩的数据还需要做数据重构，重构时还要乘于下文trans_matrix的转置，这样就抵消了正负号的影响了。

    # 4、把求解的特征值按照大小排序，提取前k个，并把对应的特征向量组成转化矩阵 形状为n*k
    index_ascending = np.argsort(eigenvalue)  # 排序并返回对应的索引，默认为升序，从小到大
    index_descending = index_ascending[::-1]  # 改成从大到小(上一步也可以直接采用-eigenvalue)
    trans_matrix = np.empty((column, 0))
    for i in range(k):
        trans_matrix = np.column_stack((trans_matrix, eigenvector[:, index_descending[i]]))  # 在列方向上增加数据
    # print('index_ascending', index_ascending)
    # print('index_descending', index_descending)
    print('4、按照特征值大小提取的前K个特征向量组成的矩阵\n', trans_matrix)

    # 5、原输入矩阵乘于转化矩阵得到输出矩阵 形状m*n * n*k = m*k
    data_out = np.dot(data_center, trans_matrix)  # <重要> 这里是中心化后的矩阵×转化矩阵。
    print('5、PCA降维后的数据矩阵：', data_out.shape)
    print('手写PCA降维后的数据矩阵：\n', data_out)

    # 6、主成分的特征值所占的比例
    eig_sum = np.sum(eigenvalue)
    eig_pca = -np.sum(np.sort(-eigenvalue)[0:k])
    eig_pca_ratio = eig_pca / eig_sum
    print('6、输入数据特征维度为{}，输出数据特征维度为{},主成分的特征值占比为{:.2%}'.format(column, k, eig_pca_ratio))

    # 对于压缩的数据还需要重构
    reconstruct = np.dot(data_out, trans_matrix.T)

    return data_out, reconstruct


if __name__ == '__main__':
    data = np.array([[10, 15, 29],
                    [15, 46, 13],
                    [23, 21, 30],
                    [11, 9, 35],
                    [42, 45, 11],
                    [9, 48, 5],
                    [11, 21, 14],
                    [8, 5, 15],
                    [11, 12, 21],
                    [21, 20, 25]])
    print('一、原始数据为\n', data)

    # 方法一、用numpy写的PCA主成分分析
    pca1, reconstruct1 = pca_detail(data, 2)
    
    # 方法二、调用sklearn的PCA方法
    pca = PCA(n_components=2)  # 相当于用sklearn的PCA创作一个实例对象，认为指定k=2
    # pca.fit(data)  # 训练数据
    pca2 = pca.fit_transform(data)  # 用训练结果对数据进行降维并输出
    print('sklearn处理后的数据形状为：', pca2.shape)
    print('sklearn处理后的特征值为：', pca.explained_variance_)  # 显示pca最后保留的特征值
    print('sklearn处理后的特征向量为：\n', pca.components_)  # 显示pca最后保留的特征向量，注意已经把列向量转化成了行向量
    print('sklearn的PCA降维后得到的数据\n', pca2)
    reconstruct2 = np.dot(pca2, pca.components_)  # 对于部分压缩的数据需要数据重构

    # 对比一下两种方法的结果，因为（部分）特征向量有正负号的问题，结果只要数据相等即可
    print('numpy写的PCA与sklearn的的PCA输出是否相同：pca1与pca2：\n', np.isclose(pca1, pca2))
    print('numpy写的PCA与sklearn的的PCA输出是否相同：pca1与-pca2：\n', np.isclose(pca1, -pca2))
    print('numpy写的PCA与sklearn的的PCA重构数据是否相同：\n', np.allclose(reconstruct1, reconstruct2))

    # 鸢尾花实例，因为降维后的数据是特征值最大的几个维度，也可以用来做分类用
    print('二、鸢尾花实例-PCA')
    x, y = load_iris(return_X_y=True)
    print('初始导入的x的形状为：', x.shape)
    print('初始导入的y的形状为：', y.shape)
    pca_iris = PCA(n_components=2)
    reduced_x = pca_iris.fit_transform(x)
    print('PCA降维后的x的形状：', reduced_x.shape)
    red_x = np.empty((0, 2))
    green_x = np.empty((0, 2))
    blue_x = np.empty((0, 2))
    for j in range(y.shape[0]):
        if y[j] == 0:
            red_x = np.row_stack((red_x, reduced_x[j]))
        if y[j] == 1:
            green_x = np.row_stack((green_x, reduced_x[j]))
        if y[j] == 2:
            blue_x = np.row_stack((blue_x, reduced_x[j]))
    # print(red_x.shape)
    plt.figure()
    plt.scatter(red_x[:, 0], red_x[:, 1], c='r', marker='+')  # 绘制散点图
    plt.scatter(green_x[:, 0], green_x[:, 1], c='g', marker='s')
    plt.scatter(blue_x[:, 0], blue_x[:, 1], c='b', marker='x')
    plt.show()
