# 最近邻插值
import cv2
import numpy as np
import matplotlib.pyplot as plt


img = 'lenna.jpeg'

def genpic_nearest(img, size):
    ori = cv2.imread(img, 0)
    h, w = ori.shape
    hratio = h / size[0]
    wratio = w / size[1]
    pic = np.zeros(size, dtype=ori.dtype)
    for i in range(size[0]):
        for j in range(size[1]):
            pic[i, j] = ori[int(i * hratio + 0.5), int(j * wratio + 0.5)]
    return pic


# newpic = genpic_nearest(img,[150,150])
# newpic
# cv2.imshow('dfw',newpic)
# cv2.waitKey(0) # ms
# cv2.destroyWindow(winname='grey')
# plt.imshow(newpic)
# cv2.imread(newpic)


def doubleline(img, size):
    ori = cv2.imread(img, 0)
    h, w = ori.shape
    hratio = (h-1) / size[0]
    wratio = (w-1) / size[1]
    print()
    pic = np.zeros(size)
    for i in range(size[0]):
        for j in range(size[1]):
            x = i * hratio
            y = j * wratio
            x0 = int(x)
            y0 = int(y)
            u = x - x0
            v = y - y0
            ij = ori[x0, y0]
            i1j = ori[x0 + 1, y0]
            ij1 = ori[x0, y0 + 1]
            i1j1 = ori[x0 + 1, y0 + 1]
            pic[i,j] = u * v * i1j1 + v * (1 - u) * ij1 + u * (1 - v) * i1j + (1 - u) * (1 - v) * ij

    return pic

newpic = doubleline(img, [2000, 2000])

plt.subplot(121)
plt.imshow(newpic)
plt.subplot(122)
plt.imshow(cv2.imread(img, 0))
plt.show()












