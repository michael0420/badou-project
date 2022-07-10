
import numpy as np
from matplotlib import pyplot as plt
import math

if __name__ == '__main__':
    # read img
    path = 'lenna.png'
    img = plt.imread(path)
    # print(img.shape)
    print(img.dtype)
    # pyplot read .png in range 0-1, float
    if path[-4:] == '.png':
        img = img * 255
    # mean for shape[-1], the result is considered as gray
    img = img.mean(axis=-1)
    plt.figure(1)
    plt.title('img gray')
    plt.imshow(img, cmap='gray')  # 此时的img_gauss是255的浮点型数据，强制类型转换才可以，gray灰阶
    plt.axis('off')

    # 高斯平滑（保留源图像的总体特征），二维高斯分布，两个方向的sigma值相等
    sigma = 0.5
    dim = int(np.round(6*sigma + 1))   # 窗口直径
    if dim % 2 == 0:  # 保证窗口为奇数
        dim += 1
    Gaussian_filter = np.zeros([dim, dim])  # 存储高斯核
    tmp = [i-dim//2 for i in range(dim)]  # 生成一个序列，即二维高斯分布自变量x, y的取值范围 [-0.5*dim, 0.5*dim]
    n1 = 1/(2*math.pi*sigma**2)  # 计算高斯核
    n2 = -1/(2*sigma**2)  # 指数中的系数
    # 计算高斯卷积和
    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i, j] = n1*math.exp(n2*(tmp[i]**2+tmp[j]**2))
    Gaussian_filter = Gaussian_filter / Gaussian_filter.sum() # 卷积核归一化，否则卷积结果会放大缩小原像素，甚至超出边界
    dx, dy = img.shape
    img_gauss = np.zeros(img.shape) # 高斯平滑后的图像
    p = dim//2 # 由于dim为奇数，p = (f-1)//2 = f//2
    img_pad = np.pad(img, ((p,p),(p,p)), 'constant') # 为了与高斯卷积核卷积而填补矩阵
    for i in range(dx):
        for j in range(dy):
            img_gauss[i,j] = np.sum(img_pad[i:i+dim, j:j+dim]*Gaussian_filter)
    print(img_gauss.dtype)
    plt.figure(2)
    plt.title('img gauss')
    plt.imshow(img_gauss, cmap='gray')  # 此时的img_gauss是255的浮点型数据，强制类型转换才可以，gray灰阶
    plt.axis('off')

    # 边缘检测，求梯度
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    gradient_x = np.zeros(img.shape) # .shape and [,] have same effect
    gradient_y = np.zeros([dx, dy])
    gradient = np.zeros(img.shape)
    img_pad = np.pad(img_gauss, ((1,1),(1,1)),'constant')
    for i in range(dx):
        for j in range(dy):
            gradient_x[i,j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_x)
            gradient_y[i,j] = np.sum(img_pad[i:i+3, j:j+3]*sobel_y)
            gradient[i,j] = np.sqrt(gradient_x[i,j]**2 + gradient_y[i,j]**2)
    gradient_x[gradient_x == 0] = 0.00000001
    angle = gradient_y / gradient_x
    plt.figure(3)
    plt.title('gradient')
    plt.imshow(gradient, cmap='gray')  # 此时的img_gauss是255的浮点型数据，强制类型转换才可以，gray灰阶
    plt.axis('off')

    # 非极大值抑制
    img_restrain = np.zeros(img.shape)
    for i in range(1, dx-1): # 不考虑最外圈
        for j in range(1, dy-1):
            temp = gradient[i-1:i+2, j-1:j+2] # 检测点周围的8个点
            if angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
                num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
            elif angle[i, j] >= 1:
                num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
            elif angle[i, j] > 0:
                num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
            elif angle[i, j] <= 0:
                num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
            # 如果当前梯度大于切线方向两边的梯度，则认为是最大值
            if gradient[i, j] > num_1 and gradient[i, j] > num_2:
                img_restrain[i, j] = gradient[i, j]
    plt.figure(4)
    plt.title('img restrain')
    plt.imshow(img_restrain, cmap='gray')  # 此时的img_gauss是255的浮点型数据，强制类型转换才可以，gray灰阶
    plt.axis('off')

    # 双阈值检测
    low_boundary = gradient.mean() * 0.5
    high_boundary = min(low_boundary * 3, 255)
    stack = []
    for i in range(1,dx-1): # 不考虑最外圈
        for j in range(1,dy-1):
            if img_restrain[i,j] >= high_boundary: # 找到强边缘并加入stack
                img_restrain[i,j] = 255
                stack.append([i,j])
            elif img_restrain[i,j] <= low_boundary: # 找到非边缘并舍弃
                img_restrain[i,j] = 0
    # 遍历所有强边缘点周围8个点，如果有弱边缘点，则将该点改为强边缘点并加入stack
    while stack:
        x, y = stack.pop()
        for dir in [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]:
            tmp_x = x + dir[0]
            tmp_y = y + dir[1]
            if low_boundary < img_restrain[tmp_x, tmp_y] < high_boundary:
                img_restrain[tmp_x, tmp_y] = 255
                stack.append([tmp_x, tmp_y])
    # 将没有转换为强边缘点的其他点都舍弃
    for i in range(dx):
        for j in range(dy):
            if img_restrain[i,j] != 0 and img_restrain[i,j] != 255:
                img_restrain[i,j] = 0
    plt.figure(5)
    plt.title('img candy')
    plt.imshow(img_restrain, cmap='gray')  # 此时的img_gauss是255的浮点型数据，强制类型转换才可以，gray灰阶
    plt.axis('off')
    plt.show()