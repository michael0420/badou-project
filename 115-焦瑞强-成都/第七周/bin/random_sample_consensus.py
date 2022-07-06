# -*- coding=utf-8 -*-

import random
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_random_sample_consensus(k, b, data_num):
    # 构造数据集
    x = np.linspace(0, 20, data_num)
    y = x * k + b
    # 添加噪声
    x_list = list()
    y_list = list()
    for i in range(data_num):
        x_list.extend([x[i] + random.uniform(-0.5, 0.5)])
        y_list.extend([y[i] + random.uniform(-0.3, 0.3)])
    # 构建离群点
    x_out_list = list()
    y_out_list = list()
    for j in range(int(data_num/3)):
        x_out_list.extend([random.uniform(0, x.max())])
        y_out_list.extend([random.uniform(b, x.max()*k+b)])

    x_list.extend(x_out_list)
    y_list.extend(y_out_list)

    # 最小二乘法求解
    A = np.vstack([np.array(x_list), np.ones(len(x_list))]).T
    y_hat_lstsq = A.dot(np.linalg.lstsq(A, np.array(y_list), rcond=None)[0])

    # 构建随机采样一致
    item_max_num = 20
    epsilon = 0.2

    m = 0
    n = 0

    inliers_num = 0
    inliers_p = 0.9

    s = 0
    while s < item_max_num:
        # 得到模型估计的点
        d1, d2 = random.sample(range(len(x_list)), 2)
        x1, x2 = x_list[d1], x_list[d2]
        y1, y2 = y_list[d1], y_list[d2]

        # 估计参数
        k = (y2 - y1) / (x2 - x1)
        b = y2 - k * x2

        # 统计内点数量
        df_data = pd.DataFrame({"x": x_list, "y": y_list})
        df_data["dis"] = df_data.apply(lambda r: np.power(np.abs(k*r["x"] + b - r["y"])/np.sqrt(k*k + 1), 2), axis=1)
        err = df_data["dis"].sum()/df_data.shape[0]
        df_inliner = df_data[df_data["dis"] < epsilon]
        cur_inliner_num = len(df_data[df_data["dis"] < epsilon])
        s = s + 1
        print("iter: {}, cur_inliner_num:{}, best_inliner_num:{}, err:{}".format(i, cur_inliner_num, inliers_num, err))

        # 更新模型参数
        if cur_inliner_num > inliers_num:
            p = cur_inliner_num / (2 * data_num)
            inliers_num = cur_inliner_num
            m = k
            n = b
            item_max_num = math.log(1 - inliers_p) / math.log(1 - np.power(p, 2))  # 更新迭代次数
            s = 0  # 当前迭代置为0
            print("[update] iter:{0}, a:{1}, b:{2}, P:{3}, best_inliner_num:{4}, iter_num:{5}".format(s, m, n,
                                                                                                      inliers_num, p,
                                                                                                      item_max_num))
        if inliers_num > data_num - int(data_num/3) - 10:
            break

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("random sample consensus")
    ax.plot(x_list, y_list, 'k.', label='data')
    y_hat = m * np.asarray(x_list) + n
    ax.plot(x_list, y_hat, color="green", label="ransac fit")
    ax.plot(x_list, y_hat_lstsq, 'r', label="line fit")
    ax.plot(x, y, 'orange', label="exact system")
    ax.plot(x_out_list, y_out_list, 'bx', label='noisy data')
    # ax.plot(df_inliner["x"], df_inliner["y"],  'k.', label="RANSAC data")
    plt.legend()
    plt.show()
