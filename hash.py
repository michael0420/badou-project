import cv2

#均值哈希
def ahash(img):
    #缩放图为8*8
    img = cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)
   #转换为灰度图
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #s为像素和初值为0，hash_str为hash值初值为''
    s=0
    hash_str=''
    #累积叠加求像素和
    for i in range(8):
        for j in range(8):
            s= s+ gray[i,j]
    #求平均灰度
    avg =s/64
    #灰度大于平均值1相反为0生成图片的hash值
    for i in range(8):
        for j in range(8):
            if gray[i,j]>avg:
                hash_str= hash_str+'1'
            else:
                hash_str = hash_str+'0'
    return hash_str

#差值算法
def dhash(img):
    #缩放成8*9的
    img = cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hash_str =''

    #像素的前一个值大于后一个则字符串加1，否则加0
    for i in range(8):
        for j in range(8):
            if gray[i,j]>gray[i,j+1]:
                hash_str = hash_str+'1'
            else:
                hash_str =hash_str+'0'

    return hash_str


def cmpHash(hash1,hash2):
    n=0
    #如果长度不同返回-1表示出错了
    if len(hash1)!= len(hash2):
        return -1

    for i in range(len(hash1)):
        if hash1[i]!=hash2[i]:
            n+=1
    return n

img1= cv2.imread("C:\\Users\\LENOVO\\Desktop\\lenna.png")
img2 =cv2.imread("C:\\Users\\LENOVO\\Desktop\\lenna_noise.png")
hash1=ahash(img1)
hash2 =ahash(img2)
print(hash1)
print(hash2)
n=cmpHash(hash1,hash2)
print("hash相似度",n)


hash1=dhash(img1)
hash2 =dhash(img2)
print(hash1)
print(hash2)
n=cmpHash(hash1,hash2)
print("hash相似度",n)



