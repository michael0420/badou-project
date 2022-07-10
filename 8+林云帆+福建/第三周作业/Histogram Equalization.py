import cv2
import numpy as np
from matplotlib import pyplot as plt

#读取图片
img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
M,N = gray.shape

#计算灰度值出现个数
H = gray.flatten().tolist()
H1 = sorted(H)
H3 = []
for x in H1 :
    if x not in H3:
        H3.append(x)
a = []
for j in H3:
    a.append(H.count(j))

#计算图片的累积直方图
cdj = []
sum = 0
for n in a:
    sum += n
    cdj.append(sum)
#print(cdj)

#进行均衡化
heq = []
for o in cdj:
    q = int((o) / (M*N) *256 -1)
    heq.append(q)


Heq = []
for p in H:
    d = H3.index(p)
    Heq.append(heq[d])

#转化到目标图上
dst = np.array(Heq,dtype=np.uint8).reshape(img.shape[0],img.shape[1])
cv2.imshow('3',np.hstack([gray,dst]))
cv2.waitKey(0)

#绘制直方图
hist = cv2.calcHist([dst],[0],None,[256],[0,256])
plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()