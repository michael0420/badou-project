import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用numpy随机生成200各随机点
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
# np.newaxis 在使用和功能上等价于 None，其实就是 None 的一个别名.可以用None代替
# 可以看到，当np.newaxis在[ , ]前面（右边）时，变为列扩展的二维数组
# 当np.newaxis在[ , ]后面（左边）时，变为行扩展的二维数组
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data)+noise


# 定义两个placeholder存放输入数据
# [None, num_input] 这表示该维度待定，你的输入是什么长度，它就是多少。
# 一般来说第一个维度是指 batch_size，而 batch_size 一般不限制死的，
# 在训练的时候可能是几十个一起训练，但是在运行模型的时候就一个一个预测，这时候 batch_size 就为 1 .
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 定义神经网络中间层
# tf.random_normal()函数用于从“服从指定正态分布的序列”中随机取出指定个数的值。
weight_l1 = tf.Variable(tf.random_normal([1, 10]))   # 权重
# 隐含层，，输出层的神经元的个数与训练数据无关，完全依靠自己设定。
# 输出层的神经元个数代表几个输出结果，本例是要求一个输入对应的输出解果，就是回归问题。
# 还有一类问题是分类，分类问题是根据训练模型，判断输入数据的类别。
bias_l1 = tf.Variable(tf.zeros([1, 10]))   # 偏置
wx_plus_b_l1 = tf.matmul(x, weight_l1) + bias_l1
L1 = tf.nn.tanh(wx_plus_b_l1)  # 激活函数

# 定义神经网络输出层

weight_l2 = tf.Variable(tf.random_normal([10, 1]))
bias_l2 = tf.Variable(tf.zeros([1, 1]))
wx_plus_b_l2 = tf.matmul(L1, weight_l2) + bias_l2
prediction = tf.nn.tanh(wx_plus_b_l2)

# 定义损失函数

loss = tf.reduce_mean(tf.square(y - prediction))  # MSE

# 定义反向传播算法
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)   # 学习率=0.1
# TensorFlow的optimizer类下的子类，属于优化器。实现的是梯度下降算法。

with tf.Session() as s:
    # 变量初始化
    s.run(tf.global_variables_initializer())
    # 训练2000次
    for i in range(2000):
        s.run(train_step, feed_dict={x: x_data, y: y_data})

    # 获得预测值
    prediction_values = s.run(prediction, feed_dict={x: x_data})

    # 画图
    plt.figure()
    plt.scatter(x_data, y_data)  # 画真实值图像
    plt.plot(x_data, prediction_values, 'r', lw=5)  # 预测值曲线
    plt.show()





