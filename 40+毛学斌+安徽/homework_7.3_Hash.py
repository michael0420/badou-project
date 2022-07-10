import numpy as np
import cv2


def aHash(img_in):
    """
    aHash:均值哈希
    输入： img_in:需要处理的图像
    输出： 均值哈希生成的哈希值
    """
    # 1.缩放：图片缩放为8 * 8，保留结构，除去细节。 
    img_in = cv2.resize(img_in, (8, 8), interpolation=cv2.INTER_LINEAR)  # 差值：双线性插值
    # 2.灰度化：转换为灰度图。
    gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    # 3.求平均值：计算灰度图所有像素的平均值。
    ave = np.mean(gray)  # 利用numpy求平均值
    # print(ave)
    # 4.比较：像素值大于平均值记作1，相反记作0，总共64位。
    # 5.生成hash：将上述步骤生成的1和0按顺序组合起来既是图片的指纹（hash）。
    hash_str = ''  # 哈希值
    for i in range(8):
        for j in range(8):
            if gray[i, j] > ave:
                hash_str += '1'   # 字符串合并
            else:
                hash_str += '0'
    # print(hash_str)
    return hash_str


def dHash(img_in):
    """
    dHash: 差值哈希
    输入： img_in:需要处理的图像
    输出： 差值哈希生成的哈希值
    """
    pass
    # 1、缩放：图片缩放为8 * 9，保留结构，除去细节。
    img_in = cv2.resize(img_in, (9, 8), interpolation=cv2.INTER_LINEAR)  # 图像形状宽*高，差值：双线性插值
    # print(img_in.shape)  # 图像形状宽*高
    # 2、灰度化：转换为灰度图。
    gray = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    # 3、比较：像素值大于后一个像素值记作1，相反记作0。本行不与下一行对比，每行9个像素， 八个差值，有8行，总共64位
    # 4、生成hash：将上述步骤生成的1和0按顺序组合起来既是图片的指纹（hash）。 6.
    hash_str = ''
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j+1]:
                hash_str += '1'
            else:
                hash_str += '0'
    return hash_str


def comHash(hash1, hash2):
    """
    对比哈希值：将两幅图的指纹（哈希值）对比，计算汉明距离，即两个64位的hash值有多少位是不一样的，不相同位数越少，图片越相似。
    输入 hash1:哈希值1
        hash2: 哈希值2
    输出： 汉明距离，二进制情况下，不一样的位数
    """
    if len(hash1) != len(hash2):
        return -1
    num = 0
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            num += 1
    return num


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    img_noise = cv2.imread('lenna_noise.png')
    a_hash_img = aHash(img)
    a_hash_noise = aHash(img_noise)
    print('均值哈希aHash    原图像的哈希值：', a_hash_img)
    print('均值哈希aHash_noise图像的哈希值：', a_hash_noise)
    print('均值哈希aHash的汉明距离，不同的位数为：', comHash(a_hash_img, a_hash_noise))
    d_hash_img = dHash(img)
    d_hash_noise = dHash(img_noise)
    print('均值哈希dHash    原图像的哈希值：', d_hash_img)
    print('均值哈希dHash_noise图像的哈希值：', d_hash_noise)
    print('均值哈希dHash的汉明距离，不同的位数为：', comHash(d_hash_img, d_hash_noise))
