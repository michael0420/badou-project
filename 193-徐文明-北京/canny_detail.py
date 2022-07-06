import cv2
import math
import numpy as np
import joblib
import matplotlib.pyplot as plt


class Canny():
    def __init__(self):
        self.pic = 'lenna.png'
        self.bgr = cv2.imread(self.pic, 1)
        self.gray = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY)


    def api_canny(self,low_threshold,high_threshold):
        cannyre = cv2.Canny(self.gray, low_threshold, high_threshold)
        return cannyre


    def show(self,x):
        cv2.imshow('canny',x)
        cv2.waitKey()
        cv2.destroyAllWindows()


    def gaussian_smooth(self,x):
        """高斯平滑"""
        sigma = 0.8 # 高斯平滑时的高斯参数，标准差
        dim = int(np.round(6 * sigma)) # 三个标准差后小于千分之三的概率，任务是可以忽略的
        if dim % 2 == 0:  # 最好是奇数,不是的话加一
            dim += 1
        Gaussian_filter = np.zeros([dim, dim])  # 存储高斯核，这是数组不是列表了
        tmp = [i - dim // 2 for i in range(dim)]  # 生成一个序列,方便计算高斯分布的值
        n1 = 1 / (2 * math.pi * sigma ** 2)  # 计算高斯核
        n2 = -1 / (2 * sigma ** 2)
        for i in range(dim):
            for j in range(dim):
                Gaussian_filter[i, j] = n1 * math.exp(n2 * (tmp[i] ** 2 + tmp[j] ** 2))
        Gaussian_filter = Gaussian_filter / Gaussian_filter.sum()
        dx, dy = x.shape
        x_new = np.zeros(x.shape)  # 存储平滑之后的图像，zeros函数得到的是浮点型数据
        tmp = dim // 2
        x_pad = np.pad(x, ((tmp, tmp), (tmp, tmp)), 'constant')  # 两个维度前后各填充半个dim
        for i in range(dx):
            for j in range(dy):
                x_new[i, j] = np.sum(x_pad[i:i + dim, j:j + dim] * Gaussian_filter) #
        x_new = x_new.astype(np.uint8)
        return x_new


    def tidu(self,x_new):
        """梯度"""
        dx,dy = x_new.shape
        sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
        tidu_x = np.zeros([dx,dy])
        tidu_y = np.zeros([dx,dy])
        tidu = np.zeros([dx,dy])
        x_new_pad = np.pad(x_new, ((1, 1), (1, 1)), 'constant')  # 边缘填补，根据上面矩阵结构3//2所以写1
        for i in range(dx):
            for j in range(dy):
                tidu_x[i, j] = np.sum(x_new_pad[i:i + 3, j:j + 3] * sobel_x)
                tidu_y[i, j] = np.sum(x_new_pad[i:i + 3, j:j + 3] * sobel_y)
                tidu[i, j] = np.sqrt(tidu_x[i, j]**2 + tidu_y[i, j]**2)
        tidu_x = np.where(tidu_x==0,0.000001,tidu_x)
        angle = tidu_y/tidu_x
        return tidu,angle


    def feizuidazhiyizhi(self,tidu,angle):
        """非极大值抑制"""
        img_yizhi = np.zeros(tidu.shape)
        dx,dy = tidu.shape
        for i in range(1, dx - 1):
            for j in range(1, dy - 1):
                flag = False  # 在8邻域内是否要抹去做个标记
                temp = tidu[i - 1:i + 2, j - 1:j + 2]  # 梯度幅值的8邻域矩阵
                if angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
                    num_1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
                    num_2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
                    if tidu[i, j] > num_1 and tidu[i, j] > num_2:
                        flag = True
                elif angle[i, j] >= 1:
                    num_1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
                    num_2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
                    if tidu[i, j] > num_1 and tidu[i, j] > num_2:
                        flag = True
                elif angle[i, j] > 0:
                    num_1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
                    num_2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
                    if tidu[i, j] > num_1 and tidu[i, j] > num_2:
                        flag = True
                elif angle[i, j] < 0:
                    num_1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
                    num_2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
                    if tidu[i, j] > num_1 and tidu[i, j] > num_2:
                        flag = True
                if flag:
                    img_yizhi[i, j] = tidu[i, j]
        return img_yizhi



    def bin_threshold(self,x,low_threshold,high_threshold):
        zhan = []
        for i in range(1, x.shape[0] - 1):  # 外圈不考虑了
            for j in range(1, x.shape[1] - 1):
                if x[i, j] >= high_threshold:  # 取，一定是边的点
                    x[i, j] = 255
                    zhan.append([i, j])
                elif x[i, j] <= low_threshold:  # 舍
                    x[i, j] = 0
        while not len(zhan) == 0:
            temp_1, temp_2 = zhan.pop()  # 出栈
            print(temp_1, temp_2,len(zhan))
            a = x[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
            if (a[0, 0] < high_threshold) and (a[0, 0] > low_threshold):
                x[temp_1 - 1, temp_2 - 1] = high_threshold  # 这个像素点标记为边缘
                zhan.append([temp_1 - 1, temp_2 - 1])  # 进栈
            if (a[0, 1] < high_threshold) and (a[0, 1] > low_threshold):
                x[temp_1 - 1, temp_2] = high_threshold
                zhan.append([temp_1 - 1, temp_2])
            if (a[0, 2] < high_threshold) and (a[0, 2] > low_threshold):
                x[temp_1 - 1, temp_2 + 1] = high_threshold
                zhan.append([temp_1 - 1, temp_2 + 1])
            if (a[1, 0] < high_threshold) and (a[1, 0] > low_threshold):
                x[temp_1, temp_2 - 1] = high_threshold
                zhan.append([temp_1, temp_2 - 1])
            if (a[1, 2] < high_threshold) and (a[1, 2] > low_threshold):
                x[temp_1, temp_2 + 1] = high_threshold
                zhan.append([temp_1, temp_2 + 1])
            if (a[2, 0] < high_threshold) and (a[2, 0] > low_threshold):
                x[temp_1 + 1, temp_2 - 1] = high_threshold
                zhan.append([temp_1 + 1, temp_2 - 1])
            if (a[2, 1] < high_threshold) and (a[2, 1] > low_threshold):
                x[temp_1 + 1, temp_2] = high_threshold
                zhan.append([temp_1 + 1, temp_2])
            if (a[2, 2] < high_threshold) and (a[2, 2] > low_threshold):
                x[temp_1 + 1, temp_2 + 1] = high_threshold
                zhan.append([temp_1 + 1, temp_2 + 1])


        x[x==high_threshold] = 255
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if x[i, j] != 0 and x[i, j] != 255:
                    x[i, j] = 0
        return x



if __name__ == '__main__':
    canny = Canny()
    # x = canny.api_canny(200,300)
    x_ori = canny.gray
    x_gaussian_smooth = canny.gaussian_smooth(canny.gray)
    x_tidu,y = canny.tidu(x_gaussian_smooth)
    x_feizuidazhiyizhi = canny.feizuidazhiyizhi(x_tidu,y)
    x_bin_threshold = canny.bin_threshold(x_feizuidazhiyizhi, 60, 100)
    # x = x.astype(np.uint8)
    # canny.show(x)

    plt.subplot(221)
    plt.imshow(x_ori,cmap='gray')
    plt.subplot(222)
    plt.imshow(x_gaussian_smooth,cmap='gray')

    plt.subplot(223)
    plt.imshow(x_tidu,cmap='gray')
    plt.subplot(224)
    plt.imshow(x_bin_threshold,cmap='gray')
    plt.show()



