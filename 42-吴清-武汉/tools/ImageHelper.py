# -*- coding:utf-8 -*-
# 八斗AI课作业.
# 图像处理工具类
# 42-吴清-武汉
# 2022-4-17
#原始lenna图尺寸372*374

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# 灰度化函数
# _filePath 图像文件路径
# 2022-4-17
def image_gray(_filePath):
    img = cv.imread(_filePath)
    print("source img:", img)
    h, w = img.shape[:2]
    img_gray = np.zeros([h, w], img.dtype)  # 创建一张指定长宽的空白图
    print("img width:%d height:%d" % (h, w))
    print("img point 0,0 channel: %d" % (len(img[0, 0])))
    for i in range(h):
        for j in range(w):
            m = img[i, j]  # 获取源图i,j的像素图
            img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)  # 给灰度图i，j点像素赋值
    print("img_gray point 0,0 ", img_gray[0, 0])
    print("img_gray:", img_gray)
    #系统方法
    #img_gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    return img_gray

# 二值化图片
# _filePath 图像文件路径
# 2022-4-17
def image_twoValue(_filePath):
    img = cv.imread(_filePath)
    print("source img:", img)
    h, w = img.shape[:2]
    img_two = np.zeros([h, w], img.dtype)  # 创建一张指定长宽的空白图

    for i in range(h):
        for j in range(w):
            m = img[i, j]  # 获取源图i,j的像素图
            img_two[i, j] = 0 if int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3) < 125 else 255  # 给二值图i，j点像素赋值
    print("img_two:", img_two)
    # 系统方法
    #img_gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #img_two=cv.threshold(img_gray,127,255,cv.THRESH_BINARY)
    return img_two

# 邻近插值算法
# _filePath 图像文件路径
# x_size 新的x尺寸
# y_size 新的y尺寸
# 2022-5-4
def image_adjacent(_filePath,x_size,y_size):
    img = cv.imread(_filePath)
    print("source img:", img)
    s_x, s_y,s_channels = img.shape # 获取原始图像的尺寸
    print("source img size:",s_x,s_y)
    img_target = np.zeros((x_size, y_size,s_channels), img.dtype)  # 创建一张新的指定长宽的空白图
    for i in range(x_size):
        for j in range(y_size):
            source_point_x=int(i*s_x/x_size) # 获取在原始图中的邻近点x坐标
            source_point_y = int(j*s_y/y_size) # 获取在原始图中的邻近点y坐标
            # print("current source point :",source_point_x,source_point_y,img[source_point_x,source_point_y])
            img_target[i,j]=img[source_point_x,source_point_y]
    print ("img_adjacent",img_target)
    return img_target

# 双线性插值算法
# _filePath 图像文件路径
# target_x 新的x尺寸
# target_y 新的y尺寸
# 2022-5-4
def image_linear(_filePath,target_x,target_y):
    img = cv.imread(_filePath)
    print("source img:", img)
    s_x, s_y,s_channels = img.shape # 获取原始图像的尺寸
    print("source img size:",s_x,s_y)
    img_target = np.zeros((target_x, target_y,s_channels), img.dtype)  # 创建一张新的指定长宽的空白图
    for c in range(3):
        for i in range(target_x):
            for j in range(target_y):
                #获取当前点在经过中心重合对齐以及缩放后在原图的位置（P点）
                point_p_x=( i+ 0.5) * s_x/target_x-0.5
                #y=(x1-x)y0+(x-x0)y1
                point_p_y = (j + 0.5) * s_y/target_y-0.5
                #获取P点在原图中邻近的4个点x0,x1,y0,y1
                point_s_x0=int(max(point_p_x,0)) #防止起点为负数
                point_s_x1=int(min(point_s_x0 + 1 ,s_x - 1)) # 防止x+1点在原图中不存在
                point_s_y0=int(max(point_p_y,0)) #防止起点为负数
                point_s_y1=int(min(point_s_y0 + 1, s_y - 1)) #防止y+1点在原图中不存在
                #根据双线性插值公式，获取当前点的当前通道值
                #f(L1)=(x1-x)f(0,0)+(x-x0)f(1,0)
                #f(L2)=(x1-x)f(0,1)+(x-x0)f(1,1)
                #f(P)=(y1-y)f(L1)+(y-y1)f(L2)
                L1= (point_s_x1 - point_p_x)*img[point_s_x0, point_s_y0, c]+ (point_p_x - point_s_x0) * img[point_s_x1, point_s_y0, c]
                L2= (point_s_x1 - point_p_x)*img[point_s_x0, point_s_y1, c]+ (point_p_x - point_s_x0) * img[point_s_x1, point_s_y1, c]
                img_target[i, j, c] = int((point_s_y1 - point_p_y) * L1 + (point_p_y - point_s_y0) * L2)
                #img_target[i, j, c] = int(img[point_s_x0,point_s_y0,c])

    print ("img_adjacent",img_target)
    return img_target

# 直方图均衡化
# _filePath 图像文件路径
# 2022-05-04
def image_histogram(_filePath):
    img_gray=image_gray(_filePath)
    h,w = img_gray.shape[:2]
    img_histogram = np.zeros([h, w], img_gray.dtype)  # 创建一张新的指定长宽的空白图
    histogramArray=[0]*256
    LUT=[0]*256
    for i in range(h):
        for j in range(w):
            grayValue=img_gray[i, j] #当前点的灰度值
            histogramArray[grayValue]=histogramArray[grayValue]+1 #记录当前灰度值的数量
    #计算累积分布直方图
    h_sum=histogramArray[0]
    for i in range(256):
        h_sum+=histogramArray[i]
        LUT[i]=255* h_sum / (w*h) #存储均衡化后的值
    for i in range(h):
        for j in range(w):
            # 均衡化处理
            img_histogram[i, j] = LUT[img_gray[i,j]]
    print("img_histogram:", img_histogram)
    return img_histogram

# Canny算法提取边缘
# _filePath 图像文件路径
# _low  阈值-低
# _high 阈值-高
# 2022-05-19
def image_canny(_filePath):
    returnImages=[]
    img=cv.imread(_filePath)
    img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY) #image_gray(_filePath)
    #returnImages.append(img_gray)
    #cv的canny算法
    #img_canny = cv.Canny(img_gray, _low, _high)

    #canny的手工算法
    # 高斯平滑
    img_gaussian= cv.GaussianBlur(img_gray,(3,3),0) #高斯滤波
    dx,dy=img_gaussian.shape
    # sobel 检测
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    img_tidu_x = np.zeros([dx, dy])  # 存储梯度图像
    img_tidu_y = np.zeros([dx, dy])
    img_tidu = np.zeros([dx, dy])
    img_pad = np.pad(img_gaussian, ((1, 1), (1, 1)), 'constant')  # 边缘填补，根据算子大小，设置为1
    # 使用sobel算子进行卷积
    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)  # x方向
            img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)  # y方向
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)# 计算梯度值
    img_tidu_x[img_tidu_x == 0] = 0.00000001
    angle = img_tidu_y / img_tidu_x
    returnImages.append(img_tidu)
    # 非极大值抑制
    img_yizhi = np.zeros(img_tidu.shape)
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
    returnImages.append(img_yizhi)

    # 双阈值检测，连接边缘。遍历所有一定是边的点,查看8邻域是否存在有可能是边的点，进栈
    h, w = img_yizhi.shape[:2]
    img_yuzhi = np.zeros([h, w], img_yizhi.dtype)
    lower_boundary = img_tidu.mean() * 0.5
    high_boundary = lower_boundary * 3  # 这里我设置高阈值是低阈值的三倍
    zhan = []
    for i in range(1, img_yizhi.shape[0] - 1):  # 外圈不考虑了
        for j in range(1, img_yizhi.shape[1] - 1):
            if img_yizhi[i, j] >= high_boundary:  # 取，一定是边的点
                img_yuzhi[i, j] = 255
                zhan.append([i, j])
            elif img_yizhi[i, j] <= lower_boundary:  # 舍
                img_yuzhi[i, j] = 0

    while not len(zhan) == 0:
        temp_1, temp_2 = zhan.pop()  # 出栈
        a = img_yuzhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
        if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
            img_yuzhi[temp_1 - 1, temp_2 - 1] = 255  # 这个像素点标记为边缘
            zhan.append([temp_1 - 1, temp_2 - 1])  # 进栈
        if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
            img_yuzhi[temp_1 - 1, temp_2] = 255
            zhan.append([temp_1 - 1, temp_2])
        if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
            img_yuzhi[temp_1 - 1, temp_2 + 1] = 255
            zhan.append([temp_1 - 1, temp_2 + 1])
        if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
            img_yuzhi[temp_1, temp_2 - 1] = 255
            zhan.append([temp_1, temp_2 - 1])
        if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
            img_yuzhi[temp_1, temp_2 + 1] = 255
            zhan.append([temp_1, temp_2 + 1])
        if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
            img_yuzhi[temp_1 + 1, temp_2 - 1] = 255
            zhan.append([temp_1 + 1, temp_2 - 1])
        if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
            img_yuzhi[temp_1 + 1, temp_2] = 255
            zhan.append([temp_1 + 1, temp_2])
        if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
            img_yuzhi[temp_1 + 1, temp_2 + 1] = 255
            zhan.append([temp_1 + 1, temp_2 + 1])

    for i in range(img_yuzhi.shape[0]):
        for j in range(img_yuzhi.shape[1]):
            if img_yuzhi[i, j] != 0 and img_yuzhi[i, j] != 255:
                img_yuzhi[i, j] = 0
    returnImages.append(img_yuzhi)

    return returnImages

# 图像的矩阵变换
# _filePath 图像文件路径
# 2022-5-25
def image_warp_martrix(_filePath):
    returnImages = []
    img = cv.imread(_filePath) #加载原始图像
    returnImages.append(img)
    trans_img=img.copy()
    src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])# 源图的四个角点
    dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]]) #源图4个点在目标图中的位置
    print(img.shape)
    # 生成透视变换矩阵；进行透视变换
    martrix = cv.getPerspectiveTransform(src, dst)
    print("warpMatrix:")
    print(martrix)
    result = cv.warpPerspective(trans_img, martrix, (337, 488))#通过变换矩阵，将源图变换成目标图
    returnImages.append(result)

    return returnImages
