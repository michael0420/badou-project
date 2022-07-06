import cv2 as c
import numpy as np
# import matplotlib.pyplot as plt

s=c.imread('guimiejiuzhu.jpg')
# c.imshow('asd',a)

print(s.shape)
r=float(input('输入缩放值(0.1-0.9)  :'))            #  可以定义缩放比例
print(r)

#计算缩放后的 目标图像 的 高和宽
dg=int(r*s.shape[0])
dk=int(r*s.shape[1])
#定义目标图像    注意  必须要有uint8   意为   八位无符号整数
d=np.zeros((dg,dk,3),dtype=np.uint8)

for i in range(dg):         #循环行和列（高和宽）
    for j in range(dk):
        si=round(i/r)       #四舍五入求 目的像素点 在原图中的坐标
        sj=round(j/r)
        d[i,j]=s[si,sj]     #赋值转移像素值

c.imshow('sss',s)
c.imshow('ddd',d)

# print(s)
# print(d)
# c.imwrite('ddd.jpg',d)
c.waitKey(0)
c.destroyAllWindows()