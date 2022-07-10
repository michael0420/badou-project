import cv2
import numpy as np


#差值算法
def dHash(img):

    img = cv2.resize(img,(9,8),interpolation=cv2.INTER_LANCZOS4)
    #img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j + 1]:
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
    imgStr1 = dHash(img1)
    imgStr2 = dHash(img2)

    print(imgStr1)
    print(imgStr2)
    n = cmpHash(imgStr1, imgStr2)
    print('均值哈希算法相似度：', n)