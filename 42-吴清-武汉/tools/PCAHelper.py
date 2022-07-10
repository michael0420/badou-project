# -*- coding:utf-8 -*-
# 八斗AI课作业.
# 特征提取工具类
# 42-吴清-武汉
# 2022-5-15

# 特征提取之PCA
# sampleArray: 样本数据array
# targetDimension:目标维度
# 2022-05-15
import numpy as np
import pandas as pd
import matplotlib.pyplot as pl

# PCA 算法实现
# 2022-5-15
# sampleArray 样本数据，样本中的一个元素代表是一个样本，有几个元素就有几个样本
def PCA(sampleArray,targetDimension):
    print("begin PCA process,sample:",sampleArray)
    # 获取样本均值，这里使用的是系统的方法，也可以使用遍历去求均值
    sample_avg = np.mean(sampleArray,0)#这里的第二个参数0即按列去平均 参数为1则按行平均 不传则汇总平均 也可以用np.mean(sampleArray.T) .T表示矩阵的倒置
    print("sample avg:",sample_avg)
    #样本中心化-零均值
    sample_center=sampleArray-sample_avg
    print("sample center:",sample_center)
    #求样本的协方差矩阵 （两种方法计算出来的协方差矩阵不一致。。。）
    #sample_covariance=np.dot(sample_center.T,sample_center)/np.shape(sample_center)[0]
    #print("sample_covariance:", sample_covariance)
    sample_covariance=np.cov(sample_center.T)
    print("sample_covariance:", sample_covariance)
    #获取协方差矩阵的特性向量和特征值
    eigen_value,eigen_vector=np.linalg.eig(sample_covariance)
    print("eigen_value:", eigen_value)
    print("eigen_vector:", eigen_vector)

    eigen_value_sorted = np.argsort(-1 * eigen_value)
    print("eigen_value_sorted:", eigen_value_sorted)
    eigen_vector_down = [eigen_vector[:, eigen_value_sorted[i]] for i in range(targetDimension)]
    sample_new_eigen = np.transpose(eigen_vector_down)
    print("sample_new_eigen:", sample_new_eigen)

    pl.figure()
    pl.plot(sample_new_eigen,'k')
    pl.xlabel("x",fontsize=16)
    pl.ylabel("y",fontsize=16)
    pl.show()
    #求原始矩阵在新的特征空间的投影
    sample_pca=np.dot(sampleArray,sample_new_eigen)
    print("sample_pca:", sample_pca)
    return sample_pca