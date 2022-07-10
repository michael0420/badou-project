# -*- coding=utf-8 -*-

import math
import numpy as np
import cv2
import matplotlib.pyplot as plt


def get_canny_edge_detection(img_path, sigma):
    ###############################################################################################################
    #                                           图像灰度化
    ###############################################################################################################
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # img = plt.imread(img_path)
    # img = img*255
    # img_gray = img.mean(axis=-1)

    ###############################################################################################################
    #                                            高斯滤波
    ###############################################################################################################
    kernel_size = int(round(6 * sigma + 1))
    kernel_size_odd = kernel_size + 1 if kernel_size & 1 == 0 else kernel_size
    high, width = img_gray.shape
    # 进行填充
    pad_dim = kernel_size_odd // 2

    # 初始化核
    gaussian_kernel = np.zeros(shape=(kernel_size_odd, kernel_size_odd), dtype=np.float64)
    # 计算核里面的值
    for i in range(kernel_size_odd):
        for j in range(kernel_size_odd):
            _i = i - pad_dim
            _j = j - pad_dim
            gaussian_kernel[i, j] = (1 / (2 * math.pi * sigma ** 2)) * math.exp(
                (_i ** 2 + _j ** 2) * (-1 / (2 * (sigma ** 2))))
    gaussian_kernel /= gaussian_kernel.sum()

    input_pad_init = np.zeros(shape=(high + 2 * pad_dim, width + 2 * pad_dim))
    input_pad_init[pad_dim: (pad_dim + high), pad_dim: (pad_dim + width)] = img_gray
    input_copy = input_pad_init.copy()
    # 进行卷积操作

    img_gaussian_filter = np.zeros((high, width))

    for h in range(high):
        for w in range(width):
            img_gaussian_filter[h, w] = np.sum(np.multiply(input_copy[h:h + kernel_size_odd, w:w + kernel_size_odd],
                                                           gaussian_kernel))
    ################################################################################################################
    #                                       图像水平、垂直与对角边缘检测
    ################################################################################################################

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    gradient_x = np.zeros((high, width))
    gradient_y = np.zeros((high, width))
    # gradient_magnitude = np.zeros(img_gaussian_filter.shape)

    img_pad = np.pad(img_gaussian_filter, ((1, 1), (1, 1)), 'constant', constant_values=(0, 0))
    for x in range(high):
        for y in range(width):
            gradient_x[x, y] = np.nansum(
                np.multiply(img_pad[x: x + sobel_x.shape[0], y: y + sobel_x.shape[1]],
                            sobel_x))

            gradient_y[x, y] = np.nansum(
                np.multiply(img_pad[x: x + sobel_y.shape[0], y: y + sobel_y.shape[1]],
                            sobel_y))
            # gradient_magnitude[x, y] = np.sqrt(gradient_x[x, y] ** 2 + gradient_y[x, y] ** 2)
    gradient_magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    gradient_x[gradient_x == 0] = 1 / float(10 ** 8)
    slope = gradient_y / gradient_x

    ###############################################################################################################
    #                                               非极大值抑制
    ###############################################################################################################
    d = gradient_magnitude.copy()
    non_maximum_suppression = np.zeros((high, width))
    for i in range(1, high - 1):
        for j in range(1, width - 1):
            if gradient_magnitude[i, j] == 0:
                non_maximum_suppression[i, j] = 0
            else:
                if abs(slope[i, j]) > 1:
                    weight = np.abs(1 / float(slope[i, j]))
                    grad_2 = d[i - 1, j]
                    grad_4 = d[i + 1, j]
                    if slope[i, j] < 0:
                        """
                       g1  g2
                           c
                           g4  g3
                       """
                        grad_1 = d[i - 1, j - 1]
                        grad_3 = d[i + 1, j + 1]
                    else:
                        """
                           g2  g1
                           c
                       g3  g4 
                       """
                        grad_1 = d[i - 1, j + 1]
                        grad_3 = d[i + 1, j - 1]
                else:
                    weight = np.abs(slope[i, j])
                    grad_2 = d[i, j - 1]
                    grad_4 = d[i, j + 1]
                    if slope[i, j] > 0:
                        """
                                g3
                         g2  c  g4
                         g1
                       """
                        grad_1 = d[i + 1, j - 1]
                        grad_3 = d[i - 1, j + 1]
                    else:
                        """
                       g1
                       g2  c  g4
                              g3
                       """
                        grad_1 = d[i - 1, j - 1]
                        grad_3 = d[i + 1, j + 1]
                tmp_1 = weight * grad_1 + (1 - weight) * grad_2
                tmp_2 = weight * grad_3 + (1 - weight) * grad_4
                if d[i, j] > max([tmp_1, tmp_2]):
                    non_maximum_suppression[i, j] = d[i, j]
    ################################################################################################################
    #                                            双阈值抑制
    ################################################################################################################
    low_threshold = np.nanmedian(gradient_magnitude) * 0.9
    high_threshold = low_threshold * 5

    strong_edges_index_list = list()
    double_threshold = non_maximum_suppression

    for m in range(1, high - 1):
        for n in range(1, width - 1):
            if non_maximum_suppression[m, n] <= low_threshold:
                double_threshold[m, n] = 0
            elif non_maximum_suppression[m, n] >= high_threshold:
                double_threshold[m, n] = 255
                strong_edges_index_list.append([m, n])

    while strong_edges_index_list:
        temp_h, temp_w = strong_edges_index_list.pop()
        neighborhood = non_maximum_suppression[temp_h-1: temp_h+2, temp_w-1:temp_w+2]
        if low_threshold < neighborhood[0, 0] < high_threshold:
            double_threshold[temp_h - 1, temp_w - 1] = 255
            strong_edges_index_list.append([temp_h-1, temp_w-1])
        if low_threshold < neighborhood[0, 1] < high_threshold:
            double_threshold[temp_h - 1, temp_w] = 255
            strong_edges_index_list.append([temp_h-1, temp_w])
        if low_threshold < neighborhood[0, 2] < high_threshold:
            double_threshold[temp_h - 1, temp_w + 1] = 255
            strong_edges_index_list.append([temp_h-1, temp_w + 1])
        if low_threshold < neighborhood[1, 0] < high_threshold:
            double_threshold[temp_h, temp_w - 1] = 255
            strong_edges_index_list.append([temp_h, temp_w - 1])
        if low_threshold < neighborhood[1, 2] < high_threshold:
            double_threshold[temp_h, temp_w + 1] = 255
            strong_edges_index_list.append([temp_h, temp_w + 1])
        if low_threshold < neighborhood[2, 0] < high_threshold:
            double_threshold[temp_h + 1, temp_w - 1] = 255
            strong_edges_index_list.append([temp_h + 1, temp_w - 1])
        if low_threshold < neighborhood[2, 1] < high_threshold:
            double_threshold[temp_h + 1, temp_w] = 255
            strong_edges_index_list.append([temp_h + 1, temp_w])
        if low_threshold < neighborhood[2, 2] < high_threshold:
            double_threshold[temp_h + 1, temp_w + 1] = 255
            strong_edges_index_list.append([temp_h + 1, temp_w + 1])

    double_threshold[(double_threshold > 0) & (double_threshold < 255)] = 0

    """
    while strong_edges_index_list:
        temp_h, temp_w = strong_edges_index_list.pop()
        neighborhood = non_maximum_suppression[temp_h-1: temp_h+2, temp_w-1:temp_w+2]
        # neighborhood = np.array(neighborhood_list)
        neighborhood_index = np.argwhere((neighborhood > low_threshold) & (neighborhood < high_threshold))
        if neighborhood_index.size > 0:
            index_init = np.repeat(np.array([[temp_h - 1, temp_w - 1]]), neighborhood_index.shape[0], axis=0)
            index_map = index_init + neighborhood_index
            strong_edges_index_list.extend(index_map.tolist())
            double_threshold[index_map] = 255
    """
    return double_threshold
