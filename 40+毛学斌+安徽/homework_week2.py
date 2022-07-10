import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt


def nearest_interpolation(img_in, out_dim):
    """
    用最邻近差值法来处理数字图像大小变化的问题（像素点个数变化）
    图像（像素个数）变大或变小过程中，新图像按照比例对应到原图像中，使用靠的最近的那个点的像素值直接作为新值
    输入：
        img:数字图像
        out_dim:希望输出的图像的形状尺寸，用（w宽，h高）的形式输入
    输出：
        用最邻近差值法处理后的数字图像
    """
    h, w, channels = img_in.shape  # 用.shape给出的形状是 高*宽，即矩阵的 行*列
    h_out, w_out = out_dim[1], out_dim[0]  # 一般图像形状的说法是 宽 * 高，所以要换一下顺序
    img_out = np.zeros((h_out, w_out, channels), dtype=np.uint8)  # 输入输出的形状要相同,np.uint8为0-255
    for i in range(h_out):
        for j in range(w_out):
            h_src = round(i / h_out * h)  # 利用round的四舍五入实现最邻近点的选择
            w_src = round(j / w_out * w)  # 也可以用int(x+0.5)的方式来实现四舍五入，最近点选择。注int向下取整
            img_out[i, j] = img_in[h_src, w_src]
    return img_out


def bilinear_interpolation(img_in, out_dim):
    """
    双线性差值：数字图像变化大小时的一种处理方法
    新图像按照比例对应到原图像中，使用靠的最近的那4个点，按照距离权重计算出新的像素值
    输入：
        img:数字图像
        out_dim:希望输出的图像的形状尺寸，用（w宽，h高）的形式输入
    输出：
        用双线性差值法resize后的数字图像
    """
    h, w, channels = img_in.shape
    h_out, w_out = out_dim[1], out_dim[0]
    img_out = np.zeros((h_out, w_out, channels), dtype=np.uint8)
    for i in range(h_out):
        for j in range(w_out):
            srcY = (i + 0.5) * h / h_out - 0.5  # 中心点对齐
            srcX = (j + 0.5) * w / w_out - 0.5
            y = max(0, srcY)  # 解决中心点对齐导致的边缘点为负数的问题
            x = max(0, srcX)
            y1 = int(np.floor(y))  # np.floor的结果是浮点数，需要用int转化成整数
            y2 = int(min(np.ceil(y), h - 1))  # 解决计算的边缘点超出原图最大值的问题
            x1 = int(np.floor(x))
            x2 = int(min(np.ceil(x), w - 1))
            img_out[i, j] = (y2 - y) * ((x2 - x) * img_in[y1, x1] + (x - x1) * img_in[y1, x2]) +\
                            (y - y1) * ((x2 - x) * img_in[y2, x1] + (x - x1) * img_in[y2, x2])
    return img_out


def histogram_color(img_in):
    """
    对彩色照片绘制直方图
    :param img_in: 输入cv2的三通道的BGR数字图像
    :return: 空
    """
    chans = cv2.split(img_in)  # 按照通道切割
    colors = ('b', 'g', 'r')
    plt.figure()
    plt.title('histogram-color')  # plt不支持中文字体
    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        # cv2.calcHist把图片先转换成一个变量，然后再通过plt.plot画出来，注意：输入参数必须带[]
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()


def gray_histogram_equalization(gray_in):
    """
    将灰度图并进行直方图均衡化处理，并展现对比图
    输入：灰度图
    输出：空
    """
    plt.figure()
    plt.subplot(221)
    plt.title('img_gray')
    plt.imshow(gray_in, cmap='gray')
    plt.subplot(223)
    plt.title('histogram')
    plt.hist(gray_in.ravel(), 256)
    plt.subplot(222)
    gray_new = cv2.equalizeHist(gray_in)
    plt.title('img after equalizeHist')
    plt.imshow(gray_new, cmap='gray')
    plt.subplot(224)
    plt.title('histogram after equalization')
    plt.hist(gray_new.ravel(), 256)
    plt.show()


def color_histogram_equalization(img_in):
    """
    彩色直方图均衡化效果展示
    :param img_in: 输入的三通道彩色图像
    :return: 空
    """
    blue, green, red = cv2.split(img_in)
    colors = ('b', 'g', 'r')
    blue_new = cv2.equalizeHist(blue)
    green_new = cv2.equalizeHist(green)
    red_new = cv2.equalizeHist(red)
    img_new = cv2.merge((blue_new, green_new, red_new))
    plt.subplot(221)
    plt.title('Before equalizeHist')
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img1)
    plt.subplot(222)
    plt.title('After equalizeHist')
    img2 = cv2.cvtColor(img_new, cv2.COLOR_BGR2RGB)
    plt.imshow(img2)
    plt.subplot(223)
    for (chan, color) in zip((blue, green, red), colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        plt.plot(hist, color=color)
    plt.subplot(224)
    for(chan, color) in zip((blue_new, green_new, red_new), colors):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        plt.plot(hist, color=color)
    plt.show()
    # cv2.imshow('histogram equalization', np.hstack((img_in, img_new)))
    # cv2.waitKey(0)


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    # print('img.shape:', img.shape)

    # 作业1 最邻近差值
    # img_nearest = nearest_interpolation(img, (600, 600))
    # print('img.nearest_interpolation:', img_nearest.shape)

    # 作业2 双线性差值
    # img_bilinear = bilinear_interpolation(img, (200, 200))
    # print('img.bilinear_interpolation:', img_bilinear.shape)
    # cv2.imshow('img source', img)
    # cv2.imshow('img nearest interpolation', img_nearest)
    # cv2.imshow('img bilinear interpolation', img_bilinear)
    # cv2.waitKey(1000)

    # 作业3 双线性差值--中心点对齐的公式推导：
    '''
    假设原图尺寸： m * m，变化后图的尺寸为： n * n，同时设偏移z个单位后中心点可以重合
    原图中心点：（m - 1） / 2 + z ， 变化后图中心点为（n - 1） / 2 + z 
    原图与变化后图符合比例关系为 m / n
    0.5m - 0.5 + z == m / n * （0.5n - 0.5 + z）
    （1 - m / n）z == 0.5（mn - m） / n - 0.5（m - 1）
    （n - m）/ n * z  == 0.5（mn - m - mn + n） / n == 0.5（n - m） / n
    z == 0.5 == 1/2
    '''

    # 作业4  直方图均衡化
    # 4.1 彩色图 图像直方图
    # histogram_color(img)

    # 4.2 灰度图利用直方图均衡化处理
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray_histogram_equalization(gray)

    # 4.3 彩色图利用直方图均衡化处理
    color_histogram_equalization(img)
