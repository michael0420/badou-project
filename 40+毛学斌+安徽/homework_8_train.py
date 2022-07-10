import numpy as np


def train(data_in, target, num):
    """
    神经网络训练过程的展示函数，结合ppt的2个输入，2个隐藏层，2个输出的网络结构
    输入：
         data_in: 初始值（第一个输入，第二个输入）
         target: 目标准确值（第一个，第二个）
         num: 迭代次数
    输出：
         无输出，直接展示训练过程
    """
    # 1、参数的随机初始化
    print("输入数据为：", data_in)
    in1, in2 = data_in[0], data_in[1]
    w1 = 0.15
    w2 = 0.2
    w3 = 0.25
    w4 = 0.3
    b1 = 0.35

    w5 = 0.4
    w6 = 0.45
    w7 = 0.5
    w8 = 0.55
    b2 = 0.6

    i = 0
    rate = 0.5
    ao1 = 0
    ao2 = 0
    E_total = 0
    while i < num:
        # 2、前向传播 - 加权求和zh/zo- 激活函数ah/ao
        zh1 = w1 * in1 + w2 * in2 + b1
        zh2 = w3 * in1 + w4 * in2 + b1
        ah1 = 1 / (1 + np.exp(-zh1))
        ah2 = 1 / (1 + np.exp(-zh2))
        zo1 = w5 * ah1 + w6 * ah2 + b2
        zo2 = w7 * ah1 + w8 * ah2 + b2
        ao1 = 1 / (1 + np.exp(-zo1))
        ao2 = 1 / (1 + np.exp(-zo2))

        # 3、计算损失函数
        Eo1 = 1 / 2 * (target[0] - ao1) ** 2
        Eo2 = 1 / 2 * (target[1] - ao2) ** 2
        E_total = Eo1 + Eo2
        # print('第{}次迭代，损失函数结果为{}'.format(i+1, E_total))
        # print('预测结果为：', ao1, ao2)

        # 4、反向传播计算偏导数
        dw5 = (ao1 - target[0]) * ao1 * (1 - ao1) * ah1
        dw6 = (ao1 - target[0]) * ao1 * (1 - ao1) * ah2
        dw7 = (ao2 - target[1]) * ao2 * (1 - ao2) * ah1
        dw8 = (ao2 - target[1]) * ao2 * (1 - ao2) * ah2
        db2 = (ao1 - target[0]) * ao1 * (1 - ao1) + (ao2 - target[1]) * ao2 * (1 - ao2)
        dah1 = ((ao1 - target[0]) * ao1 * (1 - ao1) * w5 +
                (ao2 - target[1]) * ao2 * (1 - ao2) * w7)
        dah2 = ((ao1 - target[0]) * ao1 * (1 - ao1) * w6 +
                (ao2 - target[1]) * ao2 * (1 - ao2) * w8)
        dw1 = dah1 * ah1 * (1 - ah1) * in1
        dw2 = dah1 * ah1 * (1 - ah1) * in2
        dw3 = dah2 * ah2 * (1 - ah2) * in1
        dw4 = dah2 * ah2 * (1 - ah2) * in2
        db1 = dah1 * ah1 * (1 - ah1) + dah2 * ah2 * (1 - ah2)

        # 5、梯度下降更新权值参数
        w5 = w5 - rate * dw5
        w6 = w6 - rate * dw6
        w7 = w7 - rate * dw7
        w8 = w8 - rate * dw8
        b2 = b2 - rate * db2
        w1 = w1 - rate * dw1
        w2 = w2 - rate * dw2
        w3 = w3 - rate * dw3
        w4 = w4 - rate * dw4
        b1 = b1 - rate * db1

        # 6、迭代，直到满足要求
        # print('w1:{}\nw2:{}\nw3:{}\nw4:{}\nw5:{}\nw6:{}\nw7:{}\n'
        #       'w8:{}\n'.format(w1, w2, w3, w4, w5, w6, w7, w8))
        i += 1
    print('第{}次迭代，损失函数结果为{}'.format(i, E_total))
    print('预测结果为：', ao1, ao2)


if __name__ == '__main__':
    data_input = np.array([0.05, 0.10])
    data_target = np.array([0.01, 0.99])
    print('target:', data_target)
    n = 100
    train(data_input, data_target, n)
