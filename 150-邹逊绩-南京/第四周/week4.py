# coding=utf8

import math

import cv2
import matplotlib.pyplot as plt
import numpy as np

lowThreshold0 = 0
MaxOfLow = 100
lowThreshold1 = 100
MaxOfHigh = 500
ratio = 3
kernel_size = 3

delta = 0.5


def main():
    pass


def relate(a, b, k):
    n = 0
    sum1 = np.zeros((k, k))

    for m in range(k):
        for n in range(k):
            sum1[m, n] = a[m, n] * b[m, n]
    return sum(sum(sum1))


def canny_detail(img):
    # 高斯滤波
    k = round(3 * delta) * 2 + 1
    print('模的大小为:', k)
    H = np.zeros((k, k))
    k1 = (k - 1) / 2
    for i in range(k):
        for j in range(k):
            H[i, j] = (1 / (2 * 3.14 * (delta ** 2))) * math.exp((-(i - k1) ** 2 - (j - k1) ** 2) / (2 * delta ** 2))
    k3 = [k, H]

    k = k3[0]
    H = k3[1]
    k1 = (k - 1) / 2
    [a, b] = img.shape
    k1 = int(k1)
    new1 = np.zeros((k1, b))
    new2 = np.zeros(((a + (k - 1)), k1))
    imag1 = np.r_[new1, img]
    imag1 = np.r_[imag1, new1]
    imag1 = np.c_[new2, imag1]
    imag1 = np.c_[imag1, new2]
    imgnew = np.zeros((a, b))
    sum2 = sum(sum(H))
    for i in range(k1, (k1 + a)):
        for j in range(k1, (k1 + b)):
            imgnew[(i - k1), (j - k1)] = relate(imag1[(i - k1):(i + k1 + 1), (j - k1):(j + k1 + 1)], H, k) / sum2

    plt.figure(1)
    plt.imshow(imgnew.astype(np.uint8), cmap='gray')

    # 边缘检测 网上找来的 效果感觉不错
    x = cv2.Sobel(imgnew, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(imgnew, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    plt.figure(2)
    plt.imshow(dst.astype(np.uint8), cmap='gray')

    # nms
    img_tidu = dst
    [dx, dy] = img_tidu.shape
    img_yizhi = np.zeros(img_tidu.shape)
    angle = y / x

    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            flag = True  # 在8邻域内是否要抹去做个标记
            temp = img_tidu[i - 1:i + 2, j - 1:j + 2]  # 梯度幅值的8邻域矩阵
            if angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            elif angle[i, j] < 0:
                num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
                if not (img_tidu[i, j] > num_1 and img_tidu[i, j] > num_2):
                    flag = False
            if flag:
                img_yizhi[i, j] = img_tidu[i, j]
    plt.figure(3)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')

    # 双阈值
    theta = 0

    Threshold1 = 0.000001
    Threshold2 = 0
    weight, height = np.shape(img_yizhi)
    hest = np.zeros([256], dtype=np.int32)
    for row in range(weight):
        for col in range(height):
            pv = int(img_yizhi[row, col])
            hest[pv] += 1
    tempg = -1
    N_blackground = 0
    N_object = 0
    N_all = weight * height
    for i in range(256):
        N_object += hest[i]
        for k in range(i, 256, 1):
            N_blackground += hest[k]
        for j in range(i, 256, 1):
            gSum_object = 0
            gSum_middle = 0
            gSum_blackground = 0

            N_middle = N_all - N_object - N_blackground
            w0 = N_object / N_all
            w2 = N_blackground / N_all
            w1 = 1 - w0 - w2
            for k in range(i):
                gSum_object += k * hest[k]
            u0 = gSum_object / N_object
            for k in range(i + 1, j, 1):
                gSum_middle += k * hest[k]
            u1 = gSum_middle / (N_middle + theta)

            for k in range(j + 1, 256, 1):
                gSum_blackground += k * hest[k]
            u2 = gSum_blackground / (N_blackground + theta)

            u = w0 * u0 + w1 * u1 + w2 * u2
            # print(u)
            g = w0 * (u - u0) * (u - u0) + w1 * (u - u1) * (u - u1) + w2 * (u - u2) * (u - u2)
            if tempg < g:
                tempg = g
                Threshold1 = i
                Threshold2 = j
            N_blackground -= hest[j]

        h, w = np.shape(img_yizhi)
        img_last = np.zeros([h, w], np.uint8)
        for row in range(h):
            for col in range(w):
                if img_yizhi[row, col] > Threshold2:
                    img_last[row, col] = 255
                elif img_yizhi[row, col] <= Threshold1:
                    img_last[row, col] = 0
                else:
                    img_last[row, col] = 126
        BlackgroundNum = 0
        AllNum = weight * height
        for i in range(weight):
            for j in range(height):
                if img_last[i, j] == 0:
                    BlackgroundNum += 1
        BlackgroundRatio = BlackgroundNum / AllNum
        if BlackgroundRatio < 0.4:  # 背景占比过少时，做一个反向操作
            w, h = np.shape(img_yizhi)
            for i in range(w):
                for j in range(h):
                    img_last[i, j] = 255 - img[i, j]

    plt.figure(4)
    plt.imshow(img_last.astype(np.uint8), cmap='gray')

    plt.axis('off')
    plt.show()


def setLow(low):
    lowThreshold0 = low
    canny_threshold(lowThreshold0, lowThreshold1)


def sethigh(high):
    if high < MaxOfLow:
        return
    lowThreshold1 = high
    canny_threshold(lowThreshold0, lowThreshold1)


def cannyTrack(img):
    cv2.namedWindow('canny demo')

    # 设置调节杠,
    cv2.createTrackbar('Min threshold', 'canny demo', lowThreshold0, MaxOfLow, setLow)
    cv2.createTrackbar('Max threshold', 'canny demo', lowThreshold1, MaxOfHigh, sethigh)

    canny_threshold(lowThreshold0, lowThreshold1)  # initialization

    if cv2.waitKey(0) == 27:  # wait for ESC key to exit cv2
        cv2.destroyAllWindows()


def canny_threshold(low, high):
    if high < low:
        return
    detected_edges = cv2.GaussianBlur(gray, (3, 3), 0)  # 高斯滤波
    detected_edges = cv2.Canny(detected_edges, low, high, apertureSize=kernel_size)  # 边缘检测
    dst = cv2.bitwise_and(img, img, mask=detected_edges)  # 用原始颜色添加到检测的边缘上
    cv2.imshow('canny demo', dst)


def pca():
    x = np.array([2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2, 1, 1.5, 1.1])
    y = np.array([2.4, 0.7, 2.9, 2.2, 3, 2.7, 1.6, 1.1, 1.6, 0.9])

    # 获取均值
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    scaled_x = x - mean_x
    scaled_y = y - mean_y

    # 中心化
    data = [[scaled_x[i], scaled_y[i]] for i in range(len(scaled_x))]

    # 协方差
    cov = np.cov(scaled_x, scaled_y)
    # 获取特征值， 特征向量
    eig_val, eig_vec = np.linalg.eig(cov)

    # 展示
    plt.figure(1)
    plt.plot(scaled_x, scaled_y, 'o', color='blue')
    xmin, xmax = scaled_x.min(), scaled_x.max()
    ymin, ymax = scaled_y.min(), scaled_y.max()
    dx = (xmax - xmin) * 0.2
    dy = (ymax - ymin) * 0.2
    # plt.figure(2)
    plt.xlim(xmin - dx, xmax + dx)
    plt.ylim(ymin - dy, ymax + dy)
    plt.plot([eig_vec[:, 0][0], 0], [eig_vec[:, 0][1], 0], color='red')
    plt.plot([eig_vec[:, 1][0], 0], [eig_vec[:, 1][1], 0], color='red')

    # 转置
    new_data = np.transpose(np.dot(eig_vec, np.transpose(data)))
    plt.figure(2)
    plt.plot(new_data[0], new_data[1], 'x', color='green')

    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(len(eig_val))]
    # 排序
    eig_pairs.sort(reverse=True)
    # 特征降到1维
    feature = eig_pairs[0][1]
    new_data_reduced = np.transpose(np.dot(feature, np.transpose(data)))

    plt.plot(scaled_x, scaled_y, 'o', color='red')
    plt.plot([eig_vec[:, 0][0], 0], [eig_vec[:, 0][1], 0], color='red')
    plt.plot([eig_vec[:, 1][0], 0], [eig_vec[:, 1][1], 0], color='blue')
    plt.plot(new_data[:, 0], new_data[:, 1], '^', color='blue')
    plt.plot(new_data_reduced[:], [1.2] * 10, '*', color='green')


    plt.show()


if __name__ == "__main__":
    img = cv2.imread('lenna.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换彩色图像为灰度图

    # canny 处理
    # canny_detail(gray)

    # 高低阈值调节
    # cannyTrack(img)

    # pca 实现
    pca()
