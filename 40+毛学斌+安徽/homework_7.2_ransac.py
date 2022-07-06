import numpy as np
import scipy.linalg as sl
import pylab  # 类似matplotlib.pyplot的绘图工具，避免版权问题


def random_group_index(data, num):
    """
    将输入的数据提取索引，然后打散顺序，分别分组返回
    输入： data：需要分组的数据
          num：分组的个数
    输出：num个的索引集合; 剩余点的索引集合
    """
    if num < data.shape[0]:
        all_index_ = np.arange(data.shape[0])  # 获取所有的索引
        np.random.shuffle(all_index_)  # 随机打乱顺序
        num_index = all_index_[0:num]  # 切片前num个
        rest_index = all_index_[num:]  # 切片剩下的
        return num_index, rest_index
    else:
        raise ValueError('num need to less than data.shape[0]')


def model_fit(dataX, dataY):
    """用scipy的最小二乘法返回模型"""
    x, resides_, rank_, s_ = sl.lstsq(dataX, dataY)  # 加_为了避免与全局变量重复
    return x


def get_error(dataX, dataY, model):
    """计算每一个点的残差的平方和"""
    y_fit = np.dot(dataX, model)
    err_per_error = (dataY - y_fit)**2
    return err_per_error


def ransac(dataX, dataY, n, k, t, d):
    """
    随机采样一致性（random sample consensus）,根据输入的数据和模型，迭代出最适合的模型的参数
    输入：dataX:输入数据1 相当于x
         dataY：输入数据，相当于y
         n: 随机内群点的个数
         k： 迭代的次数
         t： 符合内群点的最大残差平方和
         d： 阈值，匹配较好时，在外群中匹配出来的 内群点 的最少个数
    输出：
    """

    iteration = 0  # 迭代次数初始为0
    best_fit = None  # 最佳匹配，初始为空
    best_error = np.inf  # 最佳匹配时的 残差的平方和,初始值为无限大np.inf
    best_groupIn_index = None  # 最佳匹配时的 选出来的内群点

    while iteration < k:
        # 1、在数据中随机选择几个点设定为内群
        random_in_index, rest_index = random_group_index(dataX, n)
        randon_in_x = dataX[random_in_index]  # 根据索引转化成具体的点
        randon_in_y = dataY[random_in_index]
        rest_x = dataX[rest_index]
        rest_y = dataY[rest_index]
        # 2、计算适合内群的模型e.g.y = ax + b
        maybe_model = model_fit(randon_in_x, randon_in_y)
        # 3、把其它刚才没选到的剩余的点带入刚才建立的模型中，计算是否为内群
        rest_per_error = get_error(rest_x, rest_y, maybe_model)  # 剩余点带入模型 每个点 的偏差
        # print('rest_per_error:', rest_per_error.shape)  # 这里得到的形状为（450，1）
        also_in_index = rest_index[rest_per_error[:, 0] < t]  # 小于阈值的为内群点
        also_in_x = dataX[also_in_index]  # 根据索引转化成具体的点
        also_in_y = dataY[also_in_index]
        if len(also_in_index) > d:
            # 4、记下内群数量,本函数采用的best_error最小
            better_x = np.concatenate((randon_in_x, also_in_x))  # 将所有内群点合并在一起
            better_y = np.concatenate((randon_in_y, also_in_y))
            better_fit = model_fit(better_x, better_y)  # 根据所有内群点计算模型参数
            better_per_error = get_error(better_x, better_y, better_fit)  # 求每个点的偏差
            better_error = np.mean(better_per_error)  # 求所有点偏差的平均值 这里只有一个值了
            # 6、比较哪次计算中内群数量最多, 内群最多的那次（best_error最小）所建的模型就是我们所要求的解
            if better_error < best_error:
                best_fit = better_fit
                best_error = better_error
                print(best_error)
                best_groupIn_index = np.concatenate((random_in_index, also_in_index))
        # 5、重复以上步骤,迭代一直到满足迭代次数
        iteration += 1
    if best_fit is None:  # 如果达到迭代次数后best——fit仍为空，则返回一个错误
        raise ValueError("didn't meet right model")
    else:
        return best_fit, best_groupIn_index


if __name__ == '__main__':
    # 1、生成用来计算的数据
    n_sample = 500  # 生成500个点
    n_input = 1  # x的维度
    n_output = 1  # y的维度
    x_exact = 20 * np.random.random((n_sample, n_input))  # 生成（500，1）的输入矩阵,即y=ax+b中的x
    print('a_exact:', x_exact.shape)
    fit_exact = 60 * np.random.normal(size=(n_input, n_output))  # 生成转化关系,y=ax+b中的系数b,a组成的矩阵，本案例只有a
    print('fit_exact:', fit_exact.shape)
    print('fit_exact:', fit_exact)
    y_exact = np.dot(x_exact, fit_exact)  # 矩阵相乘计算出准确的输出矩阵，即y=ax+b中的y
    print('b_exact:', y_exact.shape)
    # 添加高斯噪声
    x_noise = x_exact + np.random.normal(size=x_exact.shape)  # 增加噪声，模拟真实数据
    y_noise = y_exact + np.random.normal(size=y_exact.shape)
    # 添加外群点或者错误点
    n_groupOut = 100
    all_index = np.arange(x_noise.shape[0])  # 生成索引序列 0-499
    np.random.shuffle(all_index)  # 随机打乱顺序
    groupOut_index = all_index[0:n_groupOut]  # 组外点的索引
    # x_noise[groupOut_index] = 20 * np.random.random(size=(n_groupOut, n_input))
    y_noise[groupOut_index] = 50 * np.random.normal(size=(n_groupOut, n_output))
    # 把其中的100点换成局外点（这里与np.dot完全不同，作为外群错误点）

    # 2、直接用最小二乘法（残差的平方和最小），针对所有的数据，生成最终模型，因为噪声点可能不够准确
    fit, resides, rank, s = sl.lstsq(x_noise, y_noise)  # scipy最小二乘法函数，fit为模型，resides误差，后两个不常用
    print('scipy_fit:', fit)
    # 3、用ransac拟合模型，并返回对应内群点的索引
    ransac_fit, ransac_in_index = ransac(x_noise, y_noise, 50, 1000, 7e3, 300)
    print('ransac.fit:', ransac_fit)
    # ransac需要预先输入模型，但本函数中已经通过函数的形式调用了，所以没有通过参数输入。
    # 随机设50个定为内群，迭代1000次，阈值最大差方7e3（这是经验值），外群的点经过计算至少300个点为内群点
    # 4、画图展示一下效果
    # 散点图：原始点的/ransac迭代后的
    pylab.plot(x_noise, y_noise, 'k.', label='data')  # 散点图，k.为用点，x画叉，不加就画线
    pylab.plot(x_noise[ransac_in_index], y_noise[ransac_in_index], 'bx', label='ransac data')  # bx蓝色叉
    # 直线图-表示各种拟合的直线：真实斜率exact_fit/用scipy拟合所有数据所得的斜率/ransac拟合的斜率
    sort_index = np.argsort(x_noise[:, 0])  # 获得排序后的索引.因为是（500，1）是矩阵，所以只能根据行排序
    # print('sort_index', sort_index.shape)
    x_exactSorted = x_exact[sort_index]  # 获得排序后的数据，以便后面画直线
    # print('x_exactSorted', x_exactSorted.shape)
    pylab.plot(x_exactSorted[:, 0], np.dot(x_exactSorted, fit_exact)[:, 0], label='fit_exact')
    pylab.plot(x_exactSorted[:, 0], np.dot(x_exactSorted, fit)[:, 0], label='scipy.lstsq-all data-fit')
    pylab.plot(x_exactSorted[:, 0], np.dot(x_exactSorted, ransac_fit)[:, 0], label='ransac-fit')
    pylab.legend()  # 显示标签
    pylab.show()
