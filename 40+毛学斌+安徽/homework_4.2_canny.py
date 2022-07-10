import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来显示中文字体
plt.rcParams['axes.unicode_minus'] = False  # 用来显示负号，使用时中文内容前加u


def canny_detail(img_in, low_threshold, high_threshold):
    """
    完整推导的canny算法，主要用于提取图片边缘，为下一步的识别分类做准备
    输入：
        img_in:输入的图像,彩色图
        low_threshold:双阈值控制部分的低阈值
        high_threshold:双阈值控制部分的高阈值
    输出：
        含有提取出来的边缘的图片，通常输出都是灰度图或者二值图
    """
    # 1、对图像进行灰度化
    gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    print('1、灰度化后图像形状：', gray.shape)
    plt.figure()
    plt.subplot(231)
    plt.title(u'1、灰度化后')
    plt.imshow(gray, cmap='gray')

    # 2、高斯滤波处理，去除噪声
    # 2.1手写卷积核
    sigma = 0.5  # 人为设定的标准差
    dim = int(np.round(sigma * 6 + 1))  # 6sigma计算卷积核的维度
    if dim % 2 == 0:
        dim = dim + 1  # 将维度调整成奇数，以符合卷积核的惯例
    temp_list = [i - dim // 2 for i in range(dim)]
    # 以下为根据高斯的公式计算卷积核的具体值
    gaussian_filter = np.zeros((dim, dim))
    num1 = 1 / (2 * math.pi * sigma ** 2)  # 高斯系数1
    num2 = -1 / (2 * sigma ** 2)  # 高斯系数2
    for i in range(dim):
        for j in range(dim):
            gaussian_filter[i, j] = num1 * math.exp(num2 * (temp_list[i]**2 + temp_list[j]**2))
    gaussian_filter = gaussian_filter / gaussian_filter.sum()
    print('高斯卷积核的形状为：', gaussian_filter.shape)
    # 2.2将原数据加padding以便后面得到same的矩阵
    half = dim // 2
    gray_padding = np.pad(gray, ((half, half), (half, half)), 'constant')  # 给原图像加padding，中间是形状，constant默认为0
    # 2.3加padding后的矩阵遍历相应的相同大小的区域 与卷积核进行元素级逐个相乘（*乘）再求和
    img_new = np.zeros(gray.shape)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            img_new[i, j] = np.sum(gray_padding[i:i+dim, j:j+dim] * gaussian_filter)  # 这一步就是卷积
    print('2、高斯滤波后的图像形状：', img_new.shape)

    plt.subplot(232)
    plt.title(u'2、高斯滤波之后')
    # print(img_new.dtype)  # 查看数据类型，下一步用.astype转变数据类型
    plt.imshow(img_new.astype(np.uint8), cmap='gray')

    # 3、（提取边缘）利用sobel等算法来提取图像中水平、垂直和对角的边缘，然后再求斜率
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sobel_x = np.zeros(img_new.shape)
    sobel_y = np.zeros(img_new.shape)
    sobel_diagonal = np.zeros(img_new.shape)
    img_padding = np.pad(img_new, ((1, 1), (1, 1)), 'constant')  # 因为是3*3卷积核所有4个角度padding各加1
    for i in range(img_new.shape[0]):
        for j in range(img_new.shape[1]):
            sobel_x[i, j] = np.sum(img_padding[i:i+3, j:j+3] * sobel_kernel_x)  # 水平卷积
            sobel_y[i, j] = np.sum(img_padding[i:i+3, j:j+3] * sobel_kernel_y)  # 垂直卷积
            sobel_diagonal[i, j] = math.sqrt(sobel_x[i, j]**2 + sobel_y[i, j]**2)  # 对角线卷积结果
    sobel_x[sobel_x == 0] = 0.00000001  # 下面求斜率时，为了避免分母为0
    angle = sobel_y / sobel_x  # 后面梯度方向上最大值抑制要用
    print('3、边缘提取后的图像形状', sobel_diagonal.shape)
    print('斜率形状为', angle.shape)
    plt.subplot(233)
    plt.title(u'3、sobel提取边缘')
    plt.imshow(sobel_diagonal, cmap='gray')

    # 4、对梯度幅值进行非极大值抑制（只保留区域内梯度变化最大的，删除附近的小点，以提取出强边缘）
    img_yizhi = np.zeros(sobel_diagonal.shape)
    for i in range(1, sobel_diagonal.shape[0]-1):
        for j in range(1, sobel_diagonal.shape[1]-1):
            temp = sobel_diagonal[i-1:i+2, j-1:j+2]  # 形成含8个邻点的九宫格
            flag = True
            if angle[i, j] <= -1:
                dtmp1 = -(temp[0, 0]-temp[0, 1])/angle[i, j] + temp[0, 1]  # 具体计算参照ppt，就是线性插值
                dtmp2 = -(temp[2, 2]-temp[2, 1])/angle[i, j] + temp[2, 1]
                if not (sobel_diagonal[i, j] > dtmp1 and sobel_diagonal[i, j] > dtmp2):  # 梯度方向上看是否为最大值
                    flag = False
            if angle[i, j] >= 1:
                dtmp1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                dtmp2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                if not (sobel_diagonal[i, j] > dtmp1 and sobel_diagonal[i, j] > dtmp2):
                    flag = False
            if angle[i, j] < 0:
                dtmp1 = -(temp[0, 0]-temp[1, 0])/angle[i, j] + temp[1, 0]
                dtmp2 = -(temp[2, 2]-temp[1, 2])/angle[i, j] + temp[1, 2]
                if not (sobel_diagonal[i, j] > dtmp1 and sobel_diagonal[i, j] > dtmp2):
                    flag = False
            if angle[i, j] > 0:
                dtmp1 = (temp[0, 2] - temp[1, 2]) / angle[i, j] + temp[1, 2]
                dtmp2 = (temp[2, 0] - temp[1, 0]) / angle[i, j] + temp[1, 0]
                if not (sobel_diagonal[i, j] > dtmp1 and sobel_diagonal[i, j] > dtmp2):
                    flag = False
            if flag:
                img_yizhi[i, j] = sobel_diagonal[i, j]
    print('4、最大值抑制后的图像形状', img_yizhi.shape)
    plt.subplot(234)
    plt.title(u'4、非极大值抑制后')
    plt.imshow(img_yizhi, cmap='gray')

    # 5、用双阈值算法来检测连接边缘（边缘分类，保留强边缘，去除弱边缘，通过中间边缘点分类来连接强边缘)
    # 5.1区分强边缘点，非边缘点，并分别设为 255，0，同时将强边缘点入站
    zhan = []
    for i in range(1, img_yizhi.shape[0]-1):
        for j in range(1, img_yizhi.shape[1]-1):
            if img_yizhi[i, j] >= high_threshold:  # 强边缘
                img_yizhi[i, j] = 255  # 强边缘设为255
                zhan.append([i, j])  # 把强边缘的索引入站，下一步根据强边缘找弱边缘
            if img_yizhi[i, j] <= low_threshold:  # 非边缘点
                img_yizhi[i, j] = 0  # 非边缘点设置为0
    # 5.2从强边缘点出发，查找附近8邻域的弱边缘点，并将其改为强边缘并入站
    while not len(zhan) == 0:
        n, m = zhan.pop()
        temp = img_yizhi[n-1:n+2, m-1:m+2]  # 左闭右开
        if (temp[0, 0] < high_threshold) and (temp[0, 0] > low_threshold):  # 确认是否为弱边缘点
            img_yizhi[n-1, m-1] = 255  # 强边缘附近8领域的弱边缘设置为强边缘
            zhan.append([n-1, m-1])  # 并作为强边缘继续搜寻附近的弱边缘点
        if (temp[0, 1] < high_threshold) and (temp[0, 1] > low_threshold):
            img_yizhi[n-1, m] = 255
            zhan.append([n-1, m])
        if (temp[0, 2] < high_threshold) and (temp[0, 2] > low_threshold):
            img_yizhi[n-1, m+1] = 255
            zhan.append([n-1, m+1])
        if (temp[1, 0] < high_threshold) and (temp[1, 0] > low_threshold):
            img_yizhi[n, m-1] = 255
            zhan.append([n, m-1])
        if (temp[1, 2] < high_threshold) and (temp[1, 2] > low_threshold):
            img_yizhi[n, m+1] = 255
            zhan.append([n, m+1])
        if (temp[2, 0] < high_threshold) and (temp[2, 0] > low_threshold):
            img_yizhi[n+1, m-1] = 255
            zhan.append([n+1, m-1])
        if (temp[2, 1] < high_threshold) and (temp[2, 1] > low_threshold):
            img_yizhi[n+1, m] = 255
            zhan.append([n+1, m])
        if (temp[2, 2] < high_threshold) and (temp[2, 2] > low_threshold):
            img_yizhi[n+1, m+1] = 255
            zhan.append([n+1, m+1])
        # 5.3将剩余的点弱边缘点改为0
        for i in range(0, img_yizhi.shape[0]):
            for j in range(0, img_yizhi.shape[1]):
                if img_yizhi[i, j] != 255 and img_yizhi[i, j] != 0:
                    img_yizhi[i, j] = 0
        print('5、双阈值检测后的图像形状', img_yizhi.shape)
        plt.subplot(235)
        plt.title('5、双阈值检测后的图像')
        plt.imshow(img_yizhi, cmap='gray')
        # plt.axis('off')  # 关掉坐标轴
        plt.show()

        return img_yizhi


def canny_lowThreshold(low):
    """
    对输入的图像进行canny提取边缘，并进行图像与运算，作为后期调节杆的回调函数，展示对比效果用
    因为带有调节杆的只有lowThreshold实时变化，其余的就只能通过调用全局变量来实现。因此本函数大量使用全局变量。
    输入：
        low：最低阈值
    输出：
        无具体返回值，主要生成图像（可以展示对比差异）
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel_edges = cv2.GaussianBlur(gray, (3, 3), 1)  # 高斯滤波，3*3卷积核,最后一个值为标准差
    canny_edges = cv2.Canny(sobel_edges, low, low*radio, apertureSize=kernel_size)  # 最后一个参数指定卷积核尺寸
    dst = cv2.bitwise_and(img, img, mask=canny_edges)  # 在原图像img上通过mask掩码图显示对比效果，相当于原图上叠加一幅图
    cv2.imshow('canny demo', dst)


if __name__ == '__main__':
    img = cv2.imread('./lenna.png')

    # 方法一，调用自己写的接口
    img_canny = canny_detail(img, 22, 66)
    cv2.imshow('canny_detail展示图像', img_canny)

    # 方法二、调用cv2.Canny接口
    cv2.imshow('cv2.Canny展示图像', cv2.Canny(img, 200, 300))
    # cv2.waitKey()

    # 方法三、带调节杆的canny效果展示
    lowThreshold = 0  # 初始低阈值设为0
    max_lowThreshold = 100
    radio = 3  # 高阈值为低阈值的三倍
    kernel_size = 3  # 卷积核的尺寸

    cv2.namedWindow('canny demo')  # 由图片创建一个面板窗口，以便下面调节杆函数调用
    cv2.createTrackbar('min Threshold', 'canny demo', lowThreshold, max_lowThreshold, canny_lowThreshold)
    '''
    设置调节杆函数，cv2.createTrackbar()
    共有5个参数，其实这五个参数看变量名就大概能知道是什么意思了
    第一个参数，是这个trackbar对象的名字
    第二个参数，是这个trackbar对象所在面板的名字
    第三个参数，是这个trackbar的默认初始值,也是调节的对象
    第四个参数，是这个trackbar上调节的范围(0~count)，也即上限
    第五个参数，是调节trackbar时调用的回调函数名
    '''
    canny_lowThreshold(lowThreshold)  # 实例化
    # 因为带有调节杆的只有lowThreshold实时变化，其余的就只能通过调用全局变量来实现
    if cv2.waitKey() == 27:  # cv2.waitKey() == 27就是代表按了ESC键
        cv2.destroyAllWindows()  # 结合上一句就是按ESC退出
