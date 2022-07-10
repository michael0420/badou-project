# -*- coding=utf-8 -*-

import numpy as np


def get_principal_component(X, components_num):
    # 特征中心化处理
    feature_zero_mean_scale = X - np.nanmean(X, axis=0)
    # 0 均值特征的协方差
    zero_mean_feature_cov = np.cov(feature_zero_mean_scale, rowvar=False, bias=True)
    # 计算协方差阵的特征值和特征向量
    cov_eigen_values_vector = np.linalg.eig(zero_mean_feature_cov)
    # 降维矩阵
    reduce_dimensionality_matrix = cov_eigen_values_vector[1][:,
                                   np.argsort(cov_eigen_values_vector[0])[::-1][:components_num]]
    # 获取主成分
    components = np.dot(feature_zero_mean_scale, reduce_dimensionality_matrix)
    return components

