import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lenna.png')

#定义绘制直方图的函数(包含灰度和彩色图像两种方案)
def get_hist(img):
    if len(img.shape) == 2:#处理灰度图
        plt.figure(figsize=(7,5))
        # plt.plot(x,y,'b-',linewidth=1, label='channal_Gray')
        plt.hist(img.ravel(),bins = 256,rwidth=0.9, range=(0,256))
        plt.xlabel('Pix_value',fontsize='8')
        plt.ylabel('Pix_num',fontsize='8')
        plt.title('Gray_hist',fontsize='8')
        plt.legend()
        plt.show()
    elif len(img.shape) == 3:#处理彩色图
        img_B, img_G, img_R = cv2.split(img)
        # img_list = [img_B,img_G,img_R]
        # color_style_list = ['b-','g-','r-']
        # channal_list = ['channal_B','channal_G','channal_R']
        plt.figure(figsize=(10, 10))
        plt.subplot(221)
        plt.hist(img_B.ravel(),bins = 256,rwidth=0.9, range=(0,256))
        plt.xlabel('Pix_value',fontsize='8')
        plt.ylabel('Pix_num',fontsize='8')
        plt.title('channal_B_hist',fontsize='8')
        plt.subplot(222)
        plt.hist(img_G.ravel(), bins=256, rwidth=0.9, range=(0, 256))
        plt.xlabel('Pix_value',fontsize='8')
        plt.ylabel('Pix_num',fontsize='8')
        plt.title('channal_G_hist',fontsize='8')
        plt.subplot(223)
        plt.hist(img_R.ravel(), bins=256, rwidth=0.9, range=(0, 256))
        plt.xlabel('Pix_value',fontsize='8')
        plt.ylabel('Pix_num',fontsize='8')
        plt.title('channal_R_hist',fontsize='8')
        plt.show()
#定义直方图均衡化函数(包含灰度和彩色图像两种方案)
def Histogram_equalization(img):
    img_h,img_w = img.shape[:2]
    img_vec = img.ravel()
    hist_vec = np.ones_like(img_vec)
    pix_num = img.shape[0] * img.shape[1]
    if len(img.shape) == 2:#处理灰度图像(形状为[h,w])
        img_Gray = img
        img_vec = img.ravel() #将矩阵展成向量
        hist_vec = np.ones_like(img_vec)
        sum_Pi = 0
        for pix in range(pix_num):
            Ni = img_vec[pix]
            Pi = Ni / pix_num
            sum_Pi += Pi
            hist_vec[pix] = int(sum_Pi * 256 - 1 + 0.5)
        return hist_vec.reshape(img_h,img_w) #将向量reshape为原来形状
    elif len(img.shape) == 3:#处理彩色图像(形状为[h,w,c])
        img_new =  None
        for i in range(3):#每个通道单独处理
            img_ = img[:,:,i]
            img_vec = img_.ravel() #将多维张量展平处理
            hist_vec = np.ones_like(img_vec)
            sum_Pi = 0
            for pix in range(pix_num):
                Ni = img_vec[pix]
                Pi = Ni / pix_num
                sum_Pi += Pi
                hist_vec[pix] = int(sum_Pi * 256 - 1 + 0.5)
            if i == 0:
                img_new = hist_vec.reshape(img_h,img_w,1)
            else:
                img_new = np.concatenate((img_new,hist_vec.reshape(img_h,img_w,1)),axis=2) #单独处理完后在维度上进行拼接
        return img_new

#处理前直方图展示
get_hist(img)
#进行直方图均衡化处理
img = Histogram_equalization(img)
#处理完后直方图展示
get_hist(img)


