import numpy as np
import matplotlib.pyplot as plt
import random

# 生成样本数据
count = 20
x = np.arange(count, dtype=np.float32)
y = 2*x + 3

for i in range(count):
    x[i] += random.uniform(-1, 1)
    y[i] += random.uniform(-1, 1)

# 计算拟合直线参数
n = count
a = (n*np.sum(x*y) - np.sum(x)*np.sum(y)) / (n*np.sum(x**2) - np.sum(x)**2)
b = np.average(y) - a*np.average(x)

# 根据拟合参数生成直线坐标点数据
x1 = np.zeros(count)
y1 = np.zeros(count)
for i in range(count):
    x1[i] = i
    y1[i] = a*x1[i] + b

# 绘图
plt.scatter(x, y)
plt.plot(x1, y1)
plt.show()