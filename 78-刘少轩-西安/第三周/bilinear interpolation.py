import cv2
import numpy as np

s=cv2.imread('lenna.png')
r=float(input('请输入缩放比例():'))
sg,sk,kernnel=s.shape

print(sg,sk,kernnel)
dg,dk=int(r*sg),int(r*sk)
print(dg,dk)
d=np.zeros((dg,dk,kernnel),np.uint8)
# print(d)

for ceng in range(3):
    for dx in range(dg):
        for dy in range(dk):
            sx,sy=(dx+0.5)/r-0.5,(dy+0.5)/r-0.5  #求对应的原坐标
            # print('sx:%s,sy:%s'%(sx,sy))

            #起初求出来的原坐标是小数，我们必须得到推算过程中出现的Q11,Q12,Q21,Q22四个点的坐标，
            # 所以在x，y方向上做 向下和向上 取整  ，如下：
            sx0=int(np.floor(sx))
            sx1=int(min(sx0+1,sk-1))   #-1)
            sy0=int(np.floor(sy))
            sy1=int(min(sy0+1,sg-1))  #-1)

            #计算f1，f2和P
            v0 = (sx1 - sx) * s[sy0, sx0, ceng] + (sx - sx0) * s[sy0, sx1, ceng]
            v1 = (sx1 - sx) * s[sy1, sx0, ceng] + (sx - sx0) * s[sy1, sx1, ceng]

            #P:
            d[dy, dx, ceng] = (sy1 - sy) * v0 + (sy - sy0) * v1

print(666666666666666)
# print(d)
cv2.imshow('sss',s)
cv2.imshow('ddd',d)
cv2.waitKey(0)
cv2.destroyAllWindows()