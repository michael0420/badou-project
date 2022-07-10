import cv2
import numpy as np
import matplotlib.pyplot as plt
 
#均值哈希算法
def aHash(img,x,y):
    img=cv2.resize(img,(x,y),interpolation=cv2.INTER_CUBIC)   # 三次插值法cv2.INTER_CUBIC    重采样插值法cv2.INTER_AREA
    # img_0=cv2.resize(img,(x,y),interpolation=cv2.INTER_AREA)
    #转换为灰度图
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # gray_0=cv2.cvtColor(img_0,cv2.COLOR_BGR2GRAY)
    #s为像素和初值为0，hash_str为hash值初值为''
    s=0
    hash_str=''
    #遍历累加求像素和
    for i in range(y):
        for j in range(x):
            s=s+gray[i,j]
    #求平均灰度
    avg=s/(x*y)
    #灰度大于平均值为1相反为0生成图片的hash值
    for i in range(y):
        for j in range(x):
            if  gray[i,j]>avg:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'            
    return hash_str,gray
 
#差值算法
def dHash(img,x,y):
    #缩放8*9
    img=cv2.resize(img,(x+1,y),interpolation=cv2.INTER_CUBIC)
    #转换灰度图
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hash_str=''
    #每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(y):
        for j in range(x):
            if   gray[i,j]>gray[i,j+1]:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'
    return hash_str,gray
 
#Hash值对比
def cmpHash(hash1,hash2,x,y):
    n=0
    #hash长度不同则返回-1代表传参出错
    if len(hash1)!=len(hash2):
        return -1
    #遍历判断
    for i in range(len(hash1)):
        #不相等则n计数+1，n最终为相似度
        if hash1[i]!=hash2[i]:
            n=n+1
    return (1-(n/(x*y)))

def Calculate_similarity(x0,y0):
    img1=cv2.imread('lenna.png')
    img2=cv2.imread('lenna_noise.png')
    # 图像缩放为x*y
    x=x0
    y=y0 
    hash1,gray_1= aHash(img1,x,y)
    hash2,gray_2= aHash(img2,x,y)
    # print(hash1)
    # print(hash2)
    n=cmpHash(hash1,hash2,x,y)
    print('均值哈希算法相似度：',n)   

    hash1,gray_3= dHash(img1,x,y)
    hash2,gray_4= dHash(img2,x,y)
    # print(hash1)
    # print(hash2)
    n=cmpHash(hash1,hash2,x,y)
    print('差值哈希算法相似度：',n)

    #用来正常显示中文标签
    plt.rcParams['font.sans-serif']=['SimHei']
    #图像转换为RGB显示
    img_0 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img_1 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img_2 = np.zeros((x,y))
    for i in range(y):
        for j in range(x):
            if   gray_1[i,j]>gray_2[i,j]:
                img_2[i,j]=1
            else:
                img_2[i,j]=0
    img_3 = np.zeros((x,y))
    for i in range(y):
        for j in range(x):
            if   gray_1[i,j]>gray_2[i,j]:
                img_3[i,j]=0
            else:
                img_3[i,j]=1
    
    #显示图像
    titles = [u'原始图像', u'缩放原始图像', u'两图像相减后图像_1', u'加噪原始图像', u'缩放加噪原始图像', u'两图像相减后图像_2']  
    images = [img_0, gray_1, img_2, img_1, gray_2,img_3]  
    for i in range(6): 
        if i==1 or 2 or 4 or 5: 
            plt.subplot(2,3,i+1)
            plt.imshow(images[i], cmap ='gray')
        else:
            plt.subplot(2,3,i+1)
            plt.imshow(images[i])
        plt.title(titles[i])  
        plt.xticks([]),plt.yticks([])  
    plt.savefig('result_2.jpg')
    plt.show()

if __name__ == "__main__":
    Calculate_similarity(60,60)

