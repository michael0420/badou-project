import numpy as np
import matplotlib.pyplot as plt
import math
import cv2 as cv


def canny(image_path):
    img = cv.imread(image_path)

    # 1. 图像灰度化
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # 2. 高斯滤波
    # sigma = 1.52
    sigma = 0.5
    dim = int(np.round(6 * sigma + 1))
    if dim % 2 == 0:
        dim += 1
    gaussian_filter = np.zeros([dim, dim])
    tmp = [i-dim//2 for i in range(dim)]
    n1 = 1/(2*math.pi*sigma**2)
    n2 = -1/(2*sigma**2)
    for i in range(dim):
        for j in range(dim):
            gaussian_filter[i, j] = n1*math.exp(n2*(tmp[i]**2+tmp[j]**2))
    gaussian_filter = gaussian_filter / gaussian_filter.sum()

    dx, dy = img_gray.shape
    img_new = np.zeros(img_gray.shape)
    tmp = dim // 2
    img_pad = np.pad(img_gray, ((tmp, tmp), (tmp, tmp)), 'constant')
    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i: i+dim, j: j+dim]*gaussian_filter)
    plt.figure(1)
    plt.imshow(img_new.astype(np.uint8))
    plt.axis('off')

    # 3. 求梯度
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_grad_x = np.zeros(img_new.shape)
    img_grad_y = np.zeros([dx, dy])
    img_grad = np.zeros(img_new.shape)
    img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')
    for i in range(dx):
        for j in range(dy):
            img_grad_x[i, j] = np.sum(img_pad[i: i+3, j: j+3]*sobel_kernel_x)
            img_grad_y[i, j] = np.sum(img_pad[i: i+3, j: j+3]*sobel_kernel_y)
            img_grad[i, j] = np.sqrt(img_grad_x[i, j]**2 + img_grad_y[i, j]**2)
    img_grad_x[img_grad_x == 0] = 0.00000001
    angle = img_grad_y / img_grad_x
    plt.figure(2)
    plt.imshow(img_grad.astype(np.uint8))
    plt.axis('off')

    # 4. 非极大值抑制
    img_restrain = np.zeros(img_grad.shape)
    for i in range(1, dx-1):
        for j in range(1, dy-1):
            flag = True  # 在8邻域内是否要抹去做个标记
            temp = img_grad[i-1:i+2, j-1:j+2]  # 梯度幅值的8邻域矩阵
            if angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
                if not (img_grad[i, j] > num_1 and img_grad[i, j] > num_2):
                    flag = False
            elif angle[i, j] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                if not (img_grad[i, j] > num_1 and img_grad[i, j] > num_2):
                    flag = False
            elif angle[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
                if not (img_grad[i, j] > num_1 and img_grad[i, j] > num_2):
                    flag = False
            elif angle[i, j] < 0:
                num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
                if not (img_grad[i, j] > num_1 and img_grad[i, j] > num_2):
                    flag = False
            if flag:
                img_restrain[i, j] = img_grad[i, j]
    plt.figure(3)
    plt.imshow(img_restrain.astype(np.uint8))
    plt.axis('off')

    # 5、双阈值检测，连接边缘。遍历所有一定是边的点,查看8邻域是否存在有可能是边的点，进栈
    lower_boundary = img_grad.mean() * 0.5
    high_boundary = lower_boundary * 3  # 这里设置高阈值是低阈值的三倍
    zhan = []
    for i in range(1, img_restrain.shape[0] - 1):
        for j in range(1, img_restrain.shape[1] - 1):
            if img_restrain[i, j] >= high_boundary:  # 取一定是边的点
                img_restrain[i, j] = 255
                zhan.append([i, j])
            elif img_restrain[i, j] <= lower_boundary:  # 舍
                img_restrain[i, j] = 0

    while not len(zhan) == 0:
        temp_1, temp_2 = zhan.pop()  # 出栈
        a = img_restrain[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
        if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
            img_restrain[temp_1 - 1, temp_2 - 1] = 255  # 这个像素点标记为边缘
            zhan.append([temp_1 - 1, temp_2 - 1])  # 进栈
        if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
            img_restrain[temp_1 - 1, temp_2] = 255
            zhan.append([temp_1 - 1, temp_2])
        if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
            img_restrain[temp_1 - 1, temp_2 + 1] = 255
            zhan.append([temp_1 - 1, temp_2 + 1])
        if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
            img_restrain[temp_1, temp_2 - 1] = 255
            zhan.append([temp_1, temp_2 - 1])
        if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
            img_restrain[temp_1, temp_2 + 1] = 255
            zhan.append([temp_1, temp_2 + 1])
        if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
            img_restrain[temp_1 + 1, temp_2 - 1] = 255
            zhan.append([temp_1 + 1, temp_2 - 1])
        if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
            img_restrain[temp_1 + 1, temp_2] = 255
            zhan.append([temp_1 + 1, temp_2])
        if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
            img_restrain[temp_1 + 1, temp_2 + 1] = 255
            zhan.append([temp_1 + 1, temp_2 + 1])

    for i in range(img_restrain.shape[0]):
        for j in range(img_restrain.shape[1]):
            if img_restrain[i, j] != 0 and img_restrain[i, j] != 255:
                img_restrain[i, j] = 0

    # 绘图
    plt.figure(4)
    plt.imshow(img_restrain.astype(np.uint8), cmap='gray')
    plt.axis('off')  # 关闭坐标刻度值
    plt.show()


if __name__ == '__main__':
    pic_path = 'lena.png'
    canny(pic_path)
