import cv2
import numpy as np


#均值哈希算法
def aHash(img):

    img = cv2.resize(img,(8,8),interpolation=cv2.INTER_LANCZOS4)
    #img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    s = 0
    hash_str = ''

    s = np.sum(gray)
    print(s)

    avg = s/64
    print(avg)


    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str



def cmpHash(hash1,hash2):
    n=0
    #hash长度不同则返回-1代表传参出错
    if len(hash1)!=len(hash2):
        return -1
    #遍历判断
    for i in range(len(hash1)):
        #不相等则n计数+1，n最终为相似度
        if hash1[i]!=hash2[i]:
            n=n+1
    return n



if __name__ == "__main__":
    img1=cv2.imread('lenna.png')
    img2=cv2.imread('lenna_noise.png')
    imgStr1 = aHash(img1)
    imgStr2 = aHash(img2)

    print(imgStr1)
    print(imgStr2)
    n = cmpHash(imgStr1, imgStr2)
    print('均值哈希算法相似度：', n)