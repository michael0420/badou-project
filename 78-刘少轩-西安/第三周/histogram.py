import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

a=cv.imread('guimiejiuzhu.jpg')
gray=cv.cvtColor(a,cv.COLOR_BGR2GRAY)
# cv.imshow('asd',a)
# cv.imshow('qwe',gray)

# print(type(hista))
# print(type(histgray))
#h灰度图 直方图 的 第一式
'''
eg: plt.hist(img.ravel(),256)  |--->

plt.figure()
plt.title('first way')
plt.ylabel('pixes')
plt.xlabel('pixex value')
# plt.hist(b.ravel(),256)
# plt.hist(a.ravel(),256)
plt.plot(hista,color='g')
plt.xlim(0,256)
plt.show()
'''

#h灰度图 直方图 的 第二式

#eg:
#   imghist=cv.calcHist()
#   plt.plot(imghist)
imghist=cv.calcHist([gray],[0],None,[256],[0,256])
# imghist=a.ravel()
print(type(imghist))
print(imghist.shape)
print(imghist)
'''
hista=cv.calcHist([a],[0],None,[256],[0,256])
histgray=cv.calcHist([gray],[0],None,[256],[0,256])

plt.figure()
plt.title('second way')
plt.plot(hista)
plt.plot(histgray)
plt.show()
'''

#彩色图像的三通道直方图计算方法
channels=cv.split(a)
colors=['g','b','r']

plt.figure()
plt.title('three channels\' histogram')
plt.xlabel('piex value',color='w')
plt.ylabel('piex counts',color='black')

for [channel,color] in zip(channels,colors):
    #1--->  plt.hist(channel.ravel(),256,color=color)

    #2--->
    hista=cv.calcHist([channel],[0],None,[256],(0,256))
    plt.plot(hista,color=color)
    plt.xlim(0,256)
plt.show()


# cv.waitKey(0)
# cv.destroyAllWindows()