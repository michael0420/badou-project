#!/usr/bin/env python
# -*-coding:utf-8-*-

import numpy as np
import matplotlib.pyplot as plt
import math

'''
canny边缘检测
'''

if __name__ == "__main__":
    '''1.彩色图像灰度化'''
    pic_path = "lenna.png"
    img = plt.imread(pic_path)
    if pic_path[-4:] == ".png":  # .png图片在这里的存储格式是0到1的浮点数，所以要扩展到255再计算
        img = img * 255  # 浮点数类型
    img_gray = img.mean(axis=-1)  # 彩色图像取平均值的方法转换灰度
    plt.figure(1)
    plt.imshow(img_gray.astype(np.uint8), cmap="gray")  # 浮点型数据，强制类型转换
    plt.axis("off")  # 关闭坐标轴

    '''2.高斯滤波'''
    sigma = 0.5  # sigma为高斯平滑时的高斯核参数，标准差，可调
    dim = int(np.round(6 * sigma + 1))  # dim为根据标准差求高斯卷积核是几乘几的，也就是维度，round是四舍五入函数
    if dim % 2 == 0:  # 卷积核为偶数时加1设置为奇数
        dim = dim + 1
    Gaussian_filter = np.zeros([dim, dim])  # 创建高斯卷积核
    tmp = [i - dim // 2 for i in range(dim)]
    n1 = 1 / (2 * math.pi * sigma ** 2)
    n2 = -1 / (2 * sigma ** 2)
    for i in range(dim):  # 计算高斯核
        for j in range(dim):
            Gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()
    img_filter = np.zeros(img_gray.shape)
    dx, dy = img_gray.shape
    tmp = dim // 2
    img_pad = np.pad(img_gray, ((tmp, tmp), (tmp, tmp)), "constant")  # 边缘填充
    for i in range(dx):  # 图像高斯滤波
        for j in range(dy):
            img_filter[i, j] = np.sum(img_pad[i:i + dim, j:j + dim] * Gaussian_filter)
    plt.figure(2)
    plt.imshow(img_filter.astype(np.uint8), cmap="gray")
    plt.axis("off")

    '''3.检测图像中的水平、垂直和对角边缘'''
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_tidu_x = np.zeros(img_filter.shape)
    img_tidu_y = np.zeros([dx, dy])
    img_tidu = np.zeros(img_filter.shape)
    img_pad = np.pad(img_filter, ((1, 1), (1, 1)), "constant")
    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)  # x方向
            img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)  # x方向
            img_tidu[i, j] = math.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)  # 梯度
    img_tidu_x[img_tidu_x == 0] = 0.00000001
    tan_angel = img_tidu_y / img_tidu_x  # 梯度的角度方向
    plt.figure(3)
    plt.imshow(img_tidu.astype(np.uint8), cmap="gray")
    plt.axis("off")

    '''4.非极大值抑制'''
    img_yizhi = np.zeros(img_tidu.shape)
    for i in range(1, dx - 1):  # 要判断8邻域，故不考虑最外面一圈
        for j in range(1, dy - 1):
            flag = False  # 8邻域内是否抑制标志位(是否8邻域内最大)
            temp = img_tidu[i - 1:i + 2, j - 1:j + 2]  # 梯度的8邻域
            if tan_angel[i, j] <= -1:
                num1 = (temp[0, 1] - temp[0, 0]) / tan_angel[i, j] + temp[0, 1]
                num2 = (temp[2, 1] - temp[2, 2]) / tan_angel[i, j] + temp[2, 1]
                if (img_tidu[i, j] >= num1 and img_tidu[i, j] >= num2):
                    flag = True
            elif tan_angel[i, j] >= 1:
                num1 = (temp[0, 2] - temp[0, 1]) / tan_angel[i, j] + temp[0, 1]
                num2 = (temp[2, 0] - temp[2, 1]) / tan_angel[i, j] + temp[2, 1]
                if (img_tidu[i, j] >= num1 and img_tidu[i, j] >= num2):
                    flag = True
            elif tan_angel[i, j] > 0:
                num1 = (temp[0, 2] - temp[1, 2]) * tan_angel[i, j] + temp[1, 2]
                num2 = (temp[2, 0] - temp[1, 0]) * tan_angel[i, j] + temp[1, 0]
                if (img_tidu[i, j] >= num1 and img_tidu[i, j] >= num2):
                    flag = True
            elif tan_angel[i, j] < 0:
                num1 = (temp[1, 0] - temp[0, 0]) * tan_angel[i, j] + temp[1, 0]
                num2 = (temp[1, 2] - temp[2, 2]) * tan_angel[i, j] + temp[1, 2]
                if (img_tidu[i, j] >= num1 and img_tidu[i, j] >= num2):
                    flag = True
            if flag:
                img_yizhi[i, j] = img_tidu[i, j]
    plt.figure(4)
    plt.imshow(img_yizhi.astype(np.uint8), cmap="gray")
    plt.axis("off")

    '''双阈值检测、连接边缘'''  # 这种方式结果不一样
    # lower_boundary = img_tidu.mean()* 0.5  # 低阈值
    # high_boundary = lower_boundary * 3  # 高阈值
    # for i in range(1, img_yizhi.shape[0] - 1):
    #     for j in range(1, img_yizhi.shape[1] - 1):
    #         if img_yizhi[i, j] >= high_boundary:
    #             img_yizhi[i, j] = 255
    #         elif img_yizhi[i, j] <= lower_boundary:
    #             img_yizhi[i, j] = 0
    #         elif ((img_yizhi[i-1, j-1]>=high_boundary) or (img_yizhi[i-1, j]>=high_boundary)  or (img_yizhi[i-1, j+1]>=high_boundary) or (img_yizhi[i, j-1]>=high_boundary)
    #               or (img_yizhi[i, j+1]>=high_boundary) or (img_yizhi[i+1, j-1]>=high_boundary) or (img_yizhi[i+1, j]>=high_boundary)  or (img_yizhi[i+1, j+1]>=high_boundary)):
    #             img_yizhi[i, j] = 255
    #         else:
    #             img_yizhi[i, j] = 0

    '''5.双阈值检测、连接边缘'''
    lower_boundary = img_tidu.mean() * 0.5  # 低阈值
    high_boundary = lower_boundary * 3  # 高阈值
    zhan = []
    for i in range(1, img_yizhi.shape[0] - 1):  # 要判断8邻域，故不考虑最外面一圈
        for j in range(1, img_yizhi.shape[1] - 1):
            if img_yizhi[i, j] >= high_boundary:  # 取
                img_yizhi[i, j] = 255
                zhan.append([i, j])
            elif img_yizhi[i, j] <= lower_boundary:  # 舍
                img_yizhi[i, j] = 0

    while not (len(zhan) == 0):
        temp_1, temp_2 = zhan.pop()  # 出栈
        # a = img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
        if (img_yizhi[temp_1 - 1, temp_2 - 1] < high_boundary) and (img_yizhi[temp_1 - 1, temp_2 - 1] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2 - 1] = 255  # 标记为边缘
            zhan.append([temp_1 - 1, temp_2 - 1])  # 进栈
        if (img_yizhi[temp_1 - 1, temp_2] < high_boundary) and (img_yizhi[temp_1 - 1, temp_2] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2] = 255  # 标记为边缘
            zhan.append([temp_1 - 1, temp_2])  # 进栈
        if (img_yizhi[temp_1 - 1, temp_2 + 1] < high_boundary) and (img_yizhi[temp_1 - 1, temp_2 + 1] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2 + 1] = 255  # 标记为边缘
            zhan.append([temp_1 - 1, temp_2 + 1])  # 进栈
        if (img_yizhi[temp_1, temp_2 - 1] < high_boundary) and (img_yizhi[temp_1, temp_2 - 1] > lower_boundary):
            img_yizhi[temp_1, temp_2 - 1] = 255  # 标记为边缘
            zhan.append([temp_1, temp_2 - 1])  # 进栈
        if (img_yizhi[temp_1, temp_2 + 1] < high_boundary) and (img_yizhi[temp_1, temp_2 + 1] > lower_boundary):
            img_yizhi[temp_1, temp_2 + 1] = 255  # 标记为边缘
            zhan.append([temp_1, temp_2 + 1])  # 进栈
        if (img_yizhi[temp_1 + 1, temp_2 - 1] < high_boundary) and (img_yizhi[temp_1 + 1, temp_2 - 1] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2 - 1] = 255
            zhan.append([temp_1 + 1, temp_2 - 1])  # 进栈
        if (img_yizhi[temp_1 + 1, temp_2] < high_boundary) and (img_yizhi[temp_1 + 1, temp_2] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2] = 255
            zhan.append([temp_1 + 1, temp_2])  # 进栈
        if (img_yizhi[temp_1 + 1, temp_2 + 1] < high_boundary) and (img_yizhi[temp_1 + 1, temp_2 + 1] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2 + 1] = 255
            zhan.append([temp_1 + 1, temp_2 + 1])  # 进栈
            
    for i in range(img_yizhi.shape[0]):
        for j in range(img_yizhi.shape[1]):
            if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
                img_yizhi[i, j] = 0

    plt.figure(5)
    plt.imshow(img_yizhi.astype(np.uint8), cmap="gray")
    plt.axis("off")
    plt.show()
