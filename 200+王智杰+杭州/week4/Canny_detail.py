import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
class Canny:
    def __init__(self,img_array,sigma,kernel_size_Gauss,lowThreshold = None,highThreshold = None):
        self.img_array = img_array
        self.img_h , self.img_w = img_array.shape
        self.sigma = sigma
        self.kernel_size_Gauss = kernel_size_Gauss
        self.lowThreshold = lowThreshold
        self.highThreshold = highThreshold
        self.GaussFiltering_img = 0
        self.img_GaussFiltering = self.GaussFiltering()
        self.img_grid_X, self.img_grid_Y, self.img_grid, self.angle= self.Sobel()
        self.img_NMS = self.NMS()
        self.img_BinaryThreshold = self.BinaryThreshold()
        self.img_Canny = self.BinaryThreshold(self.lowThreshold,self.highThreshold)
    #定义高斯核函数
    def Gausskernelfunction(self,sigma,x,y):
        return (1/(2*np.pi*sigma))*np.exp(-1*(pow(x,2)+pow(y,2))/2*sigma**2)
    #定义滤波计算
    def FilterCaculation(self,array,filterkernel,position_x,position_y):
        kernel_size,_ = filterkernel.shape
        a = 1 * int(kernel_size / 2)
        array_  =array[position_y-a:position_y+a+1,position_x-a:position_x+a+1]
        return sum(np.ravel(filterkernel*array_))
    #定义高斯滤波算法
    def GaussFiltering(self):
        #np.ones([self.kernel_size_Gauss, self.kernel_size_Gauss])
        GaussFiltering_kernel = []
        n = -1*int(self.kernel_size_Gauss/2)
        for i in range(self.kernel_size_Gauss):
            m = -1 * int(self.kernel_size_Gauss / 2)
            for j in range(self.kernel_size_Gauss):
                GaussFiltering_kernel.append(self.Gausskernelfunction(self.sigma,n,m))
                m += 1
            n += 1
        GaussFiltering_kernel = np.ravel(GaussFiltering_kernel).reshape(self.kernel_size_Gauss,self.kernel_size_Gauss)
        GaussFiltering_kernel /= sum(np.ravel(GaussFiltering_kernel)) #计算给定Sigma值时高斯滤波核
        #print(GaussFiltering_kernel)
        img_GaussFiltering = np.ones([self.img_h,self.img_w],self.img_array.dtype)
        n = 1 * int(self.kernel_size_Gauss / 2)
        img_array = np.pad(self.img_array, ((n, n), (n, n)), 'constant')
        for i in range(self.img_h):
            for j in range(self.img_w):
                img_GaussFiltering[i, j] = np.sum(img_array[i:i + self.kernel_size_Gauss, j:j + self.kernel_size_Gauss] * GaussFiltering_kernel)
        for i in range(self.img_h):
            for j in range(self.img_w):
                if i<n or j<n or i>self.img_h-n-1 or j>self.img_w-n-1:
                    img_GaussFiltering[i,j] = self.img_array[i,j]
                else:
                    img_GaussFiltering[i,j] =self.FilterCaculation(self.img_array,GaussFiltering_kernel,j,i)
        #print(img_array_.shape)

        return img_GaussFiltering
    #定义Sobel算子进行边缘检测
    def Sobel(self):
        Sobel_kernel_X =np.array([[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]])
        Sobel_kernel_Y = np.array([[1, 2, 1],
                                   [0, 0, 0],
                                   [-1, -2, -1]])
        kernel_size = 3
        img_grid_X = np.ones([self.img_h, self.img_w])#计算X方向梯度
        img_grid_Y = np.ones([self.img_h, self.img_w])#计算Y方向梯度
        img_grid = np.ones([self.img_h, self.img_w])#计算总体梯度
        img_pad = np.pad(self.img_GaussFiltering,((1,1),(1,1)),'constant')
        for i in range(self.img_h):
            for j in range(self.img_w):
                img_grid_X[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * Sobel_kernel_X)  # x方向
                img_grid_Y[i, j] = np.sum(img_pad[i:i + 3, j:j + 3] * Sobel_kernel_Y)  # y方向
                img_grid[i, j] = np.sqrt(img_grid_X[i, j] ** 2 + img_grid_Y[i, j] ** 2)
        # n = 1 * int(self.kernel_size_Gauss / 2)
        # for i in range(self.img_h):
        #     for j in range(self.img_w):
        #         if i < n or j < n or i > self.img_h - n - 1 or j > self.img_w - n - 1:
        #             continue
        #         else:
        #             img_grid_X[i, j] = self.FilterCaculation(img_pad, Sobel_kernel_X, j, i)
        #             img_grid_Y[i, j] = self.FilterCaculation(img_pad, Sobel_kernel_Y, j, i)
        #             img_grid[i,j] = np.sqrt(img_grid_X[i, j]**2 + img_grid_Y[i, j]**2)
        img_grid_X[img_grid_X == 0] = 0.0000001 #防止计算角度时除数为0
        angle = img_grid_Y /  img_grid_X
        return img_grid_X,img_grid_Y,img_grid,angle
    #定义非极大值抑制函数
    def NMS(self):
        img_NMS = np.zeros(self.img_grid.shape)
        for i in range(1, self.img_w - 1):
            for j in range(1, self.img_h - 1):
                flag = True  # 在8邻域内是否要抹去做个标记
                temp = self.img_grid[i - 1:i + 2, j - 1:j + 2]  # 梯度幅值的8邻域矩阵
                if self.angle[i, j] <= -1:  # 使用线性插值法判断抑制与否
                    num_1 = (temp[0, 1] - temp[0, 0]) / self.angle[i, j] + temp[0, 1]
                    num_2 = (temp[2, 1] - temp[2, 2]) / self.angle[i, j] + temp[2, 1]
                    if not (self.img_grid[i, j] > num_1 and self.img_grid[i, j] > num_2):
                        flag = False
                elif self.angle[i, j] >= 1:
                    num_1 = (temp[0, 2] - temp[0, 1]) / self.angle[i, j] + temp[0, 1]
                    num_2 = (temp[2, 0] - temp[2, 1]) / self.angle[i, j] + temp[2, 1]
                    if not (self.img_grid[i, j] > num_1 and self.img_grid[i, j] > num_2):
                        flag = False
                elif self.angle[i, j] > 0:
                    num_1 = (temp[0, 2] - temp[1, 2]) * self.angle[i, j] + temp[1, 2]
                    num_2 = (temp[2, 0] - temp[1, 0]) * self.angle[i, j] + temp[1, 0]
                    if not (self.img_grid[i, j] > num_1 and self.img_grid[i, j] > num_2):
                        flag = False
                elif self.angle[i, j] < 0:
                    num_1 = (temp[1, 0] - temp[0, 0]) * self.angle[i, j] + temp[1, 0]
                    num_2 = (temp[1, 2] - temp[2, 2]) * self.angle[i, j] + temp[1, 2]
                    if not (self.img_grid[i, j] > num_1 and self.img_grid[i, j] > num_2):
                        flag = False
                if flag:
                    img_NMS[i, j] = self.img_grid[i, j]
        return img_NMS

    #定义双阈值检测函数
    def BinaryThreshold(self,lowthreshold = None,highthreshold = None): #双阈值检测
        img_BinaryThreshold = self.img_NMS
        if lowthreshold == None:
            lowthreshold = self.img_grid.mean() * 0.5
        if highthreshold == None:
            highthreshold = lowthreshold*3
        stack = []
        for i in range(1, img_BinaryThreshold.shape[0] - 1):  # 不考虑外圈
            for j in range(1, img_BinaryThreshold.shape[1] - 1):
                if img_BinaryThreshold[i, j] >= highthreshold:  # 取，一定是边的点
                    img_BinaryThreshold[i, j] = 255
                elif img_BinaryThreshold[i, j] <= lowthreshold:  # 舍
                    img_BinaryThreshold[i, j] = 0
                else:
                    stack.append([i, j])
        #弱边缘点根据周围的8个像素点判断是否保留为强边缘点
        while not len(stack) == 0:
            temp_1, temp_2 = stack.pop()  # 出栈
            a = img_BinaryThreshold[temp_1 - 1:temp_1 + 2, temp_2 - 1:temp_2 + 2]
            flag = False
            if (a[0, 0] > highthreshold):
                flag = True
            if (a[0, 1] > highthreshold):
                flag = True
            if (a[0, 2] > highthreshold):
                flag = True
            if (a[1, 0] > highthreshold):
                flag = True
            if (a[1, 2] > highthreshold):
                flag = True
            if (a[2, 0] > highthreshold):
                flag = True
            if (a[2, 1] > highthreshold):
                flag = True
            if (a[2, 2] > highthreshold):
                flag = True
            if flag == True:
                img_BinaryThreshold[temp_1, temp_2] = 255
            else:
                img_BinaryThreshold[temp_1, temp_2] = 0
            #将介于0-255之间的像素点转化为黑色点
        for i in range(img_BinaryThreshold.shape[0]):
            for j in range(img_BinaryThreshold.shape[1]):
                if img_BinaryThreshold[i, j] != 0 and img_BinaryThreshold[i, j] != 255:
                    img_BinaryThreshold[i, j] = 0
        return img_BinaryThreshold


if __name__ =="__main__":
    # img = cv2.imread('lenna.png')
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    pic_path = 'lenna.png'
    img = plt.imread(pic_path)
    if pic_path[-4:] == '.png':  # .png图片在这里的存储格式是0到1的浮点数，所以要扩展到255再计算
        img = img * 255  # 还是浮点数类型
    img = img.mean(axis=-1)
    canny  = Canny(img,0.5,3,10,100)
    # canny.GaussFiltering()
    # GaussFiltering_img = canny.GaussFiltering()
    # Sobel_img = canny.Sobel('Y')
    # cv2.imshow('GaussFiltering_img', canny.img_GaussFiltering)
    # cv2.imshow('Sobel_X_img',canny.img_grid_X)
    # cv2.imshow('Sobel_Y_img', canny.img_grid_Y)
    # cv2.imshow('Sobel_X+Y_img', canny.img_grid)
    # cv2.imshow('GaussFiltering_img', canny.img_GaussFiltering)
    # cv2.imshow('NMS_img.jpg',canny.img_NMS)
    # cv2.imshow('BinaryThreshold.jpg', canny.img_BinaryThreshold)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    #用cv2.imshow()显示有问题
    plt.figure(1)
    plt.imshow(canny.img_GaussFiltering.astype(np.uint8), cmap='gray')  # 此时的img_new是255的浮点型数据，强制类型转换才可以，gray灰阶
    plt.axis('off')
    plt.figure(2)
    plt.imshow(canny.img_grid.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.figure(3)
    plt.imshow(canny.img_NMS.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.figure(4)
    plt.imshow(canny.img_BinaryThreshold.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.show()
