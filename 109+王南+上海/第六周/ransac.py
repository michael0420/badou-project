import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import sys


# 定义模型
# def model(p, x):
#     k, b = p
#     return k*x + b
def model(p, x):
    k, b = p
    return k * (x ** 2) + b


# 定义残差函数
def error(p, x, y):
    return model(p, x) - y


def generate_sample(p):
    x = np.linspace(0, 10, 30)
    k, b = p
    y = model(p, x)
    # 增加随机偏移量
    x += 2 * np.random.random_sample(x.shape) - 1
    y += 2 * np.random.random_sample(x.shape) - 1
    # 增加噪音
    x = np.hstack([x, np.random.random_sample(20) * 20 - 5])
    y = np.hstack([y, np.random.random_sample(20) * 20 - 5])

    return (x, y)


'''
    参数：
        sample_array: 样本序列
        model_func: 模型函数
        random_count: 随机抽取样本点数量
        iter_num: 循环数量
        diff_threshold: 内点距离度量阈值，小于该距离可称为内点
        ins_threshold: 内点数阈值，大于该阈值判定为合格模型
    返回：
        模型参数序列, 例如[k, b]
'''


def ransac(sample_array, model_func, *, random_count, iter_num, diff_threshold, ins_threshold):
    # sample_array的索引数组
    idx = np.arange(sample_array.shape[0])
    # 当前最好模型参数
    best_p = None
    # 当前最好模型残差平方和
    best_diff_power_sum = sys.float_info.max

    # print(sample_array)
    # print(sample_array.shape)
    # print(idx)
    # print(idx.shape)
    # print(idx[0:random_count])
    for i in range(iter_num):
        # 索引数组洗牌并随机获取指定个数的随机内点
        np.random.shuffle(idx)
        ins = sample_array[idx[0:random_count], :]
        # 除去随机内点以外的其它点
        others = sample_array[idx[random_count:], :]
        # 通过拟合模型判定为内点的集合
        # ins_plus = []
        x, y = np.hsplit(ins, 2)
        # 最小二乘拟合
        p, v = leastsq(error, (0, 2), args=(x.ravel(), y.ravel()))
        # 当前模型拟合的内点总数
        ins_num = random_count
        # 当前模型拟合内点的残差平方和
        diff_power_sum = 0
        for point in others:
            diff = abs(point[1] - model(p, point[0]))
            # others中残差小于阈值即可判定为内点
            if diff < diff_threshold:
                ins_num += 1  # 统计内点数
                # ins_plus.append(point) # 记录内点
                diff_power_sum += diff ** 2  # 计算残差平方和
        # 内点数达到阈值并且残差平方和比之前的小则替换最优模型
        if ins_num > ins_threshold and diff_power_sum < best_diff_power_sum:
            # print(f"curr_best_p:{p}, ins_num:{ins_num}, diff_power_sum:{diff_power_sum}")
            best_p = p
            best_diff_power_sum = diff_power_sum

    return best_p


if __name__ == "__main__":
    x, y = generate_sample((2, 1))
    # 最小二乘法
    p1, v = leastsq(error, (0, 2), args=(x, y))
    x1 = np.linspace(0, 10, 30)
    y1 = model(p1, x1)
    # ransac
    p2 = ransac(np.array(list(zip(x, y))), model, random_count=5, iter_num=1000, diff_threshold=8, ins_threshold=25)
    x2 = np.linspace(0, 10, 30)
    if p2 is not None:
        y2 = model(p2, x2)

    plt.scatter(x, y, color="green", label="sample points")
    plt.plot(x1, y1, color="yellow", label="leastsq line")
    if p2 is not None:
        plt.plot(x2, y2, color="red", label="ransac line")
    plt.legend()
    plt.show()
