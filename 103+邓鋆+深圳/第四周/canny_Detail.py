import numpy as np
import matplotlib.pyplot as plt
import math



def do_canny(img):
    #1、处理高斯平滑
    dim = 3
    sigma = 0.5  # 高斯平滑时的高斯核参数，标准差，可调
    Gaussian_filter = np.zeros([dim, dim])

    #tmp = [i-dim//2 for i in range(dim)]  # 生成一个序列
    tmp = [-1,0,1]


    n1 = 1 / (2 * math.pi * sigma ** 2)  # 计算高斯核
    n2 = -1 / (2 * sigma ** 2)
    for i in range(dim):
        for j in range(dim):
            Gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
    dx, dy = img.shape
    img_new = np.zeros(img.shape)  # 存储平滑之后的图像，zeros函数得到的是浮点型数据
    #tmp = dim//2

    tmp = 1

    img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp)), 'constant')  # 边缘填补,  0轴填充 1 行0， 1轴填充1列0
    print('img_pad' , img_pad)

    for i in range(dx):
        for j in range(dy):
            img_new[i, j] = np.sum(img_pad[i:i + dim, j:j + dim] * Gaussian_filter)

    #plt.figure(1)
    #plt.imshow(img_new.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
    #plt.axis('off')

    #plt.show()

    # 2、求梯度
    sobel_kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    img_tidu_x = np.zeros(img_new.shape)  # 存储梯度图像
    img_tidu_y = np.zeros([dx, dy])
    img_tidu = np.zeros(img_new.shape)
    jiaodu = np.zeros(img_new.shape)

    print('img_tidu_x',img_tidu_x.shape)
    print('img_tidu_y',img_tidu_y.shape)

    img_pad = np.pad(img_new, ((1, 1), (1, 1)), 'constant')  # 边缘填补，根据上面矩阵结构所以写1

    for i in range(dx):
        for j in range(dy):
            img_tidu_x[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_x)  # x方向
            img_tidu_y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * sobel_kernel_y)  # y方向
            img_tidu[i, j] = np.sqrt(img_tidu_x[i, j] ** 2 + img_tidu_y[i, j] ** 2)   #计算梯度值
            jiaodu[i,j] = math.degrees(math.atan2(img_tidu_y[i, j],img_tidu_x[i, j]))

            if jiaodu[i,j] < 0:
                jiaodu[i, j] = jiaodu[i,j] + 360


    img_tidu_x[img_tidu_x == 0] = 0.00000001 #避免除以0 非法操作
    img_tidu_y[img_tidu_y == 0] = 0.00000001  # 避免除以0 非法操作
    #angle = img_tidu_y / img_tidu_x

    angle = img_tidu_y / img_tidu_x   #tan0
    #jiaodu = tantheta * 180/np.pi #转化为度数
   #jiaodu = np.degrees(angle)

    print('----------jiaodu:',jiaodu)
    #angle = np.arctan(img_tidu_y / img_tidu_x)

    #print('ang`le',angle)
    #plt.figure(2)
    #plt.imshow(img_tidu.astype(np.uint8), cmap='gray')
    #plt.axis('off')
  #  plt.show()


    # 3、非极大值抑制  ---- 效果不理想
    print('非极大值抑制','---' * 20)
    img_yizhi = np.zeros(img_tidu.shape)
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            flag = True  # 在8邻域内是否要抹去做个标记
            temp = img_tidu[i - 1:i + 2, j - 1:j + 2]  # 梯度幅值的8邻域矩阵
            #通过角度判断方向
            if (jiaodu[i,j] >= 0 and jiaodu[i,j] <= 45) or (jiaodu[i,j] >= -180 and jiaodu[i,j] <= -145):
                #梯度线性插值
                #gp1 = (1-angle[i,j])*temp[2, 1] + angle[i,j]*temp[2, 2]
                #gp2 = (1-angle[i,j])*temp[0, 1] + angle[i,j]*temp[0, 0]
                gp1 = (1-angle[i,j])*temp[1, 2] + angle[i,j]*temp[0, 2]
                gp2 = (1-angle[i,j])*temp[1, 0] + angle[i,j]*temp[2, 0]

                if not ((img_tidu[i, j] >= gp1) and (img_tidu[i, j] >= gp2)):
                    flag = False


            elif  (jiaodu[i,j] > 45 and jiaodu[i,j] <= 90) or (jiaodu[i,j] > -145 and jiaodu[i,j] <= -90):

                #gp1 = (1 - angle[i, j]) * temp[1, 2] + angle[i, j] * temp[2, 2]
                #gp2 = (1 - angle[i, j]) * temp[1, 0] + angle[i, j] * temp[0, 0]
                gp1 = (1 - angle[i, j]) * temp[0, 1] + angle[i, j] * temp[0, 2]
                gp2 = (1 - angle[i, j]) * temp[2, 1] + angle[i, j] * temp[2, 0]

                if not ((img_tidu[i, j] >= gp1) and (img_tidu[i, j] >= gp2)):
                    flag = False


            elif (jiaodu[i, j] > 90 and jiaodu[i,j] <= 145) or (jiaodu[i, j] > -145 and jiaodu[i, j] <= -90):

                #gp1 = (1 - angle[i, j]) * temp[1, 2] + angle[i, j] * temp[0, 2]
                #gp2 = (1 - angle[i, j]) * temp[1, 0] + angle[i, j] * temp[2, 0]
                gp1 = (1 - angle[i, j]) * temp[0, 1] + angle[i, j] * temp[0, 0]
                gp2 = (1 - angle[i, j]) * temp[2, 1] + angle[i, j] * temp[2, 2]

                if not ((img_tidu[i, j] >= gp1) and (img_tidu[i, j] >= gp2)):
                    flag = False


            elif (jiaodu[i, j] > 145 and jiaodu[i,j] <= 180) or (jiaodu[i, j] > -90 and jiaodu[i,j] < 0):

                #gp1 = (1 - angle[i, j]) * temp[0, 1] + angle[i, j] * temp[0, 2]
                #gp2 = (1 - angle[i, j]) * temp[2, 1] + angle[i, j] * temp[2, 0]
                gp1 = (1 - angle[i, j]) * temp[1, 0] + angle[i, j] * temp[0, 0]
                gp2 = (1 - angle[i, j]) * temp[1, 2] + angle[i, j] * temp[2, 2]

                if not ((img_tidu[i, j] >= gp1) and (img_tidu[i, j] >= gp2)):
                    flag = False

            if flag:
                img_yizhi[i, j] = img_tidu[i, j]

    plt.figure(3)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')

    '''
    angle = img_tidu_y / img_tidu_x
    img_yizhi = np.zeros(img_tidu.shape)
    for i in range(1, dx - 1):
        for j in range(1, dy - 1):
            flag = True  # 在8邻域内是否要抹去做个标记
            temp = img_tidu[i - 1:i + 2, j - 1:j + 2]  # 梯度幅值的8邻域矩阵
            #print(temp)
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
    plt.axis('off')
 '''

    # 4、双阈值检测，连接边缘。遍历所有一定是边的点,查看8邻域是否存在有可能是边的点，进栈
    lower_boundary = img_tidu.mean() * 0.8
    high_boundary = lower_boundary * 3  # 这里我设置高阈值是低阈值的三倍
    zhan = []
    for i in range(1, img_yizhi.shape[0] - 1):  # 外圈不考虑了
        for j in range(1, img_yizhi.shape[1] - 1):
            if img_yizhi[i, j] >= high_boundary:  # 取，一定是边的点
                img_yizhi[i, j] = 255
                zhan.append([i, j])
            elif img_yizhi[i, j] <= lower_boundary:  # 舍
                img_yizhi[i, j] = 0

    while not len(zhan) == 0:
        temp_1, temp_2 = zhan.pop()  # 出栈
        a = img_yizhi[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
        if (a[0, 0] < high_boundary) and (a[0, 0] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2 - 1] = 255  # 这个像素点标记为边缘
            zhan.append([temp_1 - 1, temp_2 - 1])  # 进栈
        if (a[0, 1] < high_boundary) and (a[0, 1] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2] = 255
            zhan.append([temp_1 - 1, temp_2])
        if (a[0, 2] < high_boundary) and (a[0, 2] > lower_boundary):
            img_yizhi[temp_1 - 1, temp_2 + 1] = 255
            zhan.append([temp_1 - 1, temp_2 + 1])
        if (a[1, 0] < high_boundary) and (a[1, 0] > lower_boundary):
            img_yizhi[temp_1, temp_2 - 1] = 255
            zhan.append([temp_1, temp_2 - 1])
        if (a[1, 2] < high_boundary) and (a[1, 2] > lower_boundary):
            img_yizhi[temp_1, temp_2 + 1] = 255
            zhan.append([temp_1, temp_2 + 1])
        if (a[2, 0] < high_boundary) and (a[2, 0] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2 - 1] = 255
            zhan.append([temp_1 + 1, temp_2 - 1])
        if (a[2, 1] < high_boundary) and (a[2, 1] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2] = 255
            zhan.append([temp_1 + 1, temp_2])
        if (a[2, 2] < high_boundary) and (a[2, 2] > lower_boundary):
            img_yizhi[temp_1 + 1, temp_2 + 1] = 255
            zhan.append([temp_1 + 1, temp_2 + 1])

    for i in range(img_yizhi.shape[0]):
        for j in range(img_yizhi.shape[1]):
            if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
                img_yizhi[i, j] = 0
   # 绘图
    plt.figure(4)
    plt.imshow(img_yizhi.astype(np.uint8), cmap='gray')
    plt.axis('off')  # 关闭坐标刻度值
    plt.show()


img_path = 'w2.png'
img = plt.imread(img_path)

if img_path[-4:] == '.png':
    img = img * 255
img = img.mean(axis=-1)  # 取均值就是灰度化了
do_canny(img)
print('img',img)
#plt.imshow(img, cmap='gray')
#plt.show()
