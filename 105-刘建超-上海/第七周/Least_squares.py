#!/usr/bin/env python
# coding=utf-8

import pandas as pd

'''最小二乘法'''
sales = pd.read_csv("train_data.csv", sep=",", engine="python")  # 读取csv
# sales=pd.read_csv('train_data.csv',sep='\s*,\s*',engine='python')  #读取CSV
X = sales["X"].values  # 读取第一列
Y = sales["Y"].values  # 读取第二列

# 初始化赋值变量
s1 = 0
s2 = 0
s3 = 0
s4 = 0
N = 4  # 数据的个数

for i in range(N):
    s1 = s1 + X[i] * Y[i]  # X*Y，求和
    s2 = s2 + X[i]  # X的和
    s3 = s3 + Y[i]  # Y的和
    s4 = s4 + X[i] * X[i]  # X**2，求和

# 求斜率和截距
k = (N * s1 - s2 * s3) / (N * s4 - s2 * s2)
b = (s3 - k * s2) / N
print("Coeff:{} Intercept:{}".format(k, b))