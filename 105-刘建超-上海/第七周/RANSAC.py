#!/usr/bin/python
# encoding=utf-8

import numpy as np
import scipy as sp
import scipy.linalg as sl

'''随机采样一致性（random sample consensus）'''


def ransac(data, model, n, k, t, d, debug=False, return_all=False):
    """
        输入:
            data - 样本点
            model - 假设模型:事先自己确定
            n - 生成模型所需的最少样本点,设置为内群的个数
            k - 最大迭代次数
            t - 阈值:作为判断点满足模型的条件
            d - 拟合较好时,需要的样本点最少的个数,当做阈值看待
        输出:
            bestfit - 最优拟合解（返回nil,如果未找到）

        iterations = 0
        bestfit = nil #后面更新
        besterr = something really large #后期更新besterr = thiserr
        while iterations < k
        {
            maybeinliers = 从样本中随机选取n个,不一定全是局内点,甚至全部为局外点
            maybemodel = n个maybeinliers 拟合出来的可能符合要求的模型
            alsoinliers = emptyset #满足误差要求的样本点,开始置空
            for (每一个不是maybeinliers的样本点)
            {
                if 满足maybemodel即error < t
                    将点加入alsoinliers
            }
            if (alsoinliers样本点数目 > d)
            {
                %有了较好的模型,测试模型符合度
                bettermodel = 利用所有的maybeinliers 和 alsoinliers 重新生成更好的模型
                thiserr = 所有的maybeinliers 和 alsoinliers 样本点的误差度量
                if thiserr < besterr
                {
                    bestfit = bettermodel
                    besterr = thiserr
                }
            }
            iterations++
        }
        return bestfit
        """
    iterations = 0  # 迭代次数变量
    bestfit = None  # 最优拟合
    besterr = np.inf  # 最小残差平方和，numpy.inf：表示+∞，是没有确切的数值的numpy
    best_inlier_idxs = None  # 内群点索引
    while iterations < k:  # 迭代K次
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])  # 获取n个内群索引、剩余索引
        print("test_idxs=", test_idxs)
        maybe_inliers = data[maybe_idxs, :]  # 获取内群点，size(maybe_idxs)行数据(Xi,Yi)
        test_points = data[test_idxs]  # 获取测试点，size(test_idxs)行数据(Xi,Yi)
        maybe_model = model.fit(maybe_inliers)  # 最小二乘法拟合模型
        test_err = model.get_error(test_points, maybe_model)  # 计算误差平方和
        print("test_err=", test_err < t)  # 打印测试数据中的点是否满足模型条件
        also_idxs = test_idxs[test_err < t]  # 测试数据中满足模型条件点的索引
        print("also_idxs=", also_idxs)
        also_inliers = data[also_idxs, :]  # 测试数据中未内群的数据点
        if debug:
            print('test_err.min()', test_err.min())
            print('test_err.max()', test_err.max())
            print('numpy.mean(test_err)', np.mean(test_err))
            print('iteration %d:len(alsoinliers) = %d' % (iterations, len(also_inliers)))
        print("d:", d)  # 拟合时需要的样本点最少的个数,当做阈值看待
        if (len(also_inliers) > d):  # 如果测试数据中内群点数量大于需要的样本点最少的个数
            betterdata = np.concatenate((maybe_inliers, also_inliers))  # 连接设定的内群点和测试数据中为内群的点
            better_model = model.fit(betterdata)  # 最小二乘法拟合模型
            better_err = model.get_error(betterdata, better_model)  # 计算每行误差平方和
            thiserr = np.mean(better_err)  # 计算每行平均误差平方和
            if (thiserr < besterr):  # 判断是否小于最小残差平方和
                bestfit = better_model  # 最优拟合
                besterr = thiserr  # 最小残差平方和
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))  # 更新内群点索引
        iterations += 1
    if bestfit is None:
        raise ValueError("did't meet fit acceptance criteria")
    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit


def random_partition(n, n_data):
    # 获取n个随机索引、剩余索引
    all_idxs = np.arange(n_data)  # 获取n_data数据索引,返回一个有终点和起点的固定步长的排列
    np.random.shuffle(all_idxs)  # 将all_idxs顺序打乱
    idxs1 = all_idxs[:n]  # 获取前n个索引
    idxs2 = all_idxs[n:]  # 获取除前n个之后的索引
    return idxs1, idxs2


class LinearLeastSquareModel:  # 最小二乘求线性解,用于RANSAC的输入模型
    def __init__(self, input_column, output_column, debug=False):
        self.input_column = input_column
        self.output_column = output_column
        self.debug = debug

    def fit(self, data):
        # np.vstack按竖直方向(行顺序)构成一个新数组
        A = np.vstack([data[:, i] for i in self.input_column]).T  # 第一列Xi-->行Xi
        B = np.vstack([data[:, i] for i in self.output_column]).T  # 第二列Yi-->行Yi
        x, resids, rank, s = sp.linalg.lstsq(A, B)  # 最小二乘法模块
        return x  # 返回最小平方和向量

    def get_error(self, data, model):
        A = np.vstack([data[:, i] for i in self.input_column]).T  # 第一列Xi-->行Xi
        B = np.vstack([data[:, i] for i in self.output_column]).T  # 第二列Yi-->行Yi
        # B_fit = sp.dot(A, model)  # 计算的y值,B_fit = model.k*A + model.b
        # B_fit = np.dot(A, model)  # 计算的y值,B_fit = model.k*A + model.b
        B_fit = A * model  # 计算y值,B_fit = model.k*A + model.b
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)  # 按照行(每行)求误差的平方和
        return err_per_point


def test():
    # 生成理想数据
    n_samples = 500  # 样本数据个数
    n_inputs = 1  # 输入变量个数
    n_outputs = 1  # 输出变量个数
    A_exact = 20 * np.random.random((n_samples, n_inputs))  # 生成500行1列的0-20之前的随机数据
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))  # 随机线性度，即随机生成一个斜率
    # B_exact = sp.dot(A_exact, perfect_fit)  # B_exact=A_exact*perfect_fit
    B_exact = np.dot(A_exact, perfect_fit)  # B_exact=A_exact*perfect_fit
    # B_exact = A_exact * perfect_fit

    # 加入高斯噪声
    A_noise = A_exact + np.random.normal(size=A_exact.shape)  # 500 * 1行向量,代表Xi
    B_noise = B_exact + np.random.normal(size=B_exact.shape)  # 500 * 1行向量,代表Yi

    if 1:
        # 添加局外点
        n_outliers = 100
        all_idxs = np.arange(A_noise.shape[0])  # 生成索引0-499
        np.random.shuffle(all_idxs)  # 打乱all_idxs顺序
        outlier_idxs = all_idxs[:n_outliers]  # 获取100个随机索引
        A_noise[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))  # 加入高斯噪声和局外点的Xi
        B_noise[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))  # 加入高斯噪声和局外点的Yi
    all_data = np.hstack((A_noise, B_noise))  # 形式([Xi,Yi]....) shape:(500,2)500行2列
    input_columns = range(n_inputs)  ##数组的第一列x:0
    output_columns = [n_inputs + i for i in range(n_outputs)]  # 数组最后一列y:1
    debug = False
    model = LinearLeastSquareModel(input_columns, output_columns, debug)  # 类的实例化:用最小二乘生成已知模型,用于RANSAC的输入模型
    # 最小二乘法模块
    linear_fit, resids, rank, s = sl.lstsq(all_data[:, input_columns], all_data[:, output_columns])
    # 执行ransac算法
    ransac_fit, ransac_idxs = ransac(all_data, model, 40, 1000, 6000, 250, debug=debug, return_all=True)

    if 1:  # 一定会执行
        import pylab

        sort_idxs = np.argsort(A_exact[:, 0])  # A_exact第一列数据按升序排列的索引
        A_col0_sorted = A_exact[sort_idxs]  # 二维数组，秩为2

        if 1:
            pylab.plot(A_noise[:, 0], B_noise[:, 0], "k.", label="data")
            # pylab.plot(A_noise[:, 0], B_noise[:, 0], "r*",label="data")
            pylab.plot(A_noise[ransac_idxs["inliers"], 0], B_noise[ransac_idxs["inliers"], 0], 'rx',
                       label="RANSAC data")
        else:
            pylab.plot(A_noisy[non_outlier_idxs, 0], B_noisy[non_outlier_idxs, 0], 'k.', label='noisy data')
            pylab.plot(A_noisy[outlier_idxs, 0], B_noisy[outlier_idxs, 0], 'r.', label='outlier data')

        pylab.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, ransac_fit)[:, 0], label="RANSAC fit")
        pylab.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, perfect_fit)[:, 0], label="exact fit")
        pylab.plot(A_col0_sorted[:, 0], np.dot(A_col0_sorted, linear_fit)[:, 0], label="linear fit")

        pylab.legend()  # 图例
        pylab.show()  # 显示


if __name__ == "__main__":
    test()
