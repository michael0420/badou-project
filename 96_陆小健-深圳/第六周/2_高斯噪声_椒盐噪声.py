#随机生成符合正态（高斯）分布的随机数，means,sigma为两个参数
#随机生成符合正态（高斯）分布的随机数，means,sigma为两个参数
import numpy as np
import cv2
from numpy import shape 
import random
import matplotlib.pyplot as plt
# 在图像m*n个像素点下，每次取一个随机点，randX 代表随机生成的行，randY代表随机生成的列
# random.randint生成随机整数
def GaussianNoise(src,means,sigma,percentage):   # 生成高斯噪声函数
    NoiseImg=src
    NoiseNum=int(percentage*src.shape[0]*src.shape[1])
    for i in range(NoiseNum):
        randX=random.randint(0,src.shape[0]-1)
        randY=random.randint(0,src.shape[1]-1)
        #此处在原有像素灰度值上加上随机数
        NoiseImg[randX,randY]=NoiseImg[randX,randY]+random.gauss(means,sigma)
        #若灰度值小于0则强制为0，若灰度值大于255则强制为255
        for i in range(3):
            if  NoiseImg[randX, randY][i]< 0:
                NoiseImg[randX, randY][i]=0
            elif NoiseImg[randX, randY][i]>255:
                NoiseImg[randX, randY][i]=255
    return NoiseImg

def  fun1(src,percetage):  # 生成椒盐噪声函数   
	NoiseImg=src    
	NoiseNum=int(percetage*src.shape[0]*src.shape[1])    
	for i in range(NoiseNum): 
		randX=random.randint(0,src.shape[0]-1)
		randY=random.randint(0,src.shape[1]-1) 
		#random.random生成随机浮点数，随意取到一个像素点有一半的可能是白点255，一半的可能是黑点0      
		if random.random()<=0.5:           
			NoiseImg[randX,randY]=0       
		else:            
			NoiseImg[randX,randY]=255    
	return NoiseImg

img_0 = cv2.imread('lenna.png',1)
img = img_0.copy()
img_1 = GaussianNoise(img,2,8,0.3)
img = img_0.copy()
img_2 = GaussianNoise(img,2,8,0.6)
img = img_0.copy()
img_3 = fun1(img,0.3)
img = img_0.copy()
img_4 = fun1(img,0.6)
#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

#图像转换为RGB显示
img_0 = cv2.cvtColor(img_0, cv2.COLOR_BGR2RGB)
img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
img_3 = cv2.cvtColor(img_3, cv2.COLOR_BGR2RGB)
img_4 = cv2.cvtColor(img_4, cv2.COLOR_BGR2RGB)
#显示图像
titles = [u'原始图像', u'高斯噪声 30%', u'高斯噪声 60%', u'椒盐噪声 30%', u'椒盐噪声 60%']  
images = [img_0, img_1, img_2, img_3, img_4]  
for i in range(5):  
   plt.subplot(2,3,i+1), plt.imshow(images[i]), 
   plt.title(titles[i])  
   plt.xticks([]),plt.yticks([])  
plt.savefig('lenna_GaussianNoise_PepperandSalt.jpg')
plt.show()

