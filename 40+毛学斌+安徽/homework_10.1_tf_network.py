import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# 1、定义op：常量、变量、操作
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]  # 等间距创建200个数，并转化为列向量
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

x = tf.compat.v1.placeholder(tf.float32, [None, 1])
y = tf.compat.v1.placeholder(tf.float32, [None, 1])

w_i2h = tf.Variable(tf.random.normal([1, 10]))  # 矩阵(200,1)*(1,10)=(200,10)
b_i2h = tf.Variable(tf.zeros([1, 10]))
hidden_in = tf.matmul(x, w_i2h) + b_i2h  # 加权求和
hidden_out = tf.nn.tanh(hidden_in)  # 激活函数

w_h2o = tf.Variable(tf.random.normal([10, 1]))  # 矩阵(200,10)*(10,1)=(200,1)
b_h2o = tf.Variable(tf.zeros([1, 1]))
final_in = tf.matmul(hidden_out, w_h2o) + b_h2o
predict = tf.tanh(final_in)

loss = tf.reduce_mean(tf.square(y - predict))
op_train = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss)
op_init = tf.compat.v1.global_variables_initializer()
# 2、session中运行,自动溯源
with tf.compat.v1.Session() as sess:
    sess.run(op_init)
    for i in range(1000):
        sess.run(op_train, feed_dict={x: x_data, y: y_data})
    y_predict = sess.run(predict, feed_dict={x: x_data})

# 3、画图展示一下效果
plt.figure()
plt.scatter(x_data, y_data)
plt.plot(x_data, y_predict, 'r-', lw=5)  # 画图r红色，-形状，lw粗细
plt.show()
