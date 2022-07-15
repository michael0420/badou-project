import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread('C:\\Users\\LENOVO\\Desktop\\lenna.png')
print(img.shape)

# 图像
data = img.reshape(-1, 3)  # 不知道几行，系统自动分，分成3列每列对应RGB
data = np.float32(data)

# 停止条件 (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 设置标签
flags = cv2.KMEANS_RANDOM_CENTERS

# K-Means 聚类，聚成2类
compactness, labels2, centers2 = cv2.kmeans(data, 2, None, criteria, 10, flags)
# K-Means 聚类，聚成4类
compactness, labels4, centers4 = cv2.kmeans(data, 4, None, criteria, 10, flags)
# K-Means 聚类，聚成8类
compactness, labels8, centers8 = cv2.kmeans(data, 8, None, criteria, 10, flags)
##K-Means 聚类，聚成16类
compactness, labels16, centers16 = cv2.kmeans(data, 16, None, criteria, 10, flags)
##K-Means 聚类，聚成64类
compactness, labels64, centers64 = cv2.kmeans(data, 64, None, criteria, 10, flags)

# 图像回转成uint8二维数据类型
centers2 = np.uint8(centers2)  # 由聚类的中心组成的数组，分成两类
res = centers2[labels2.flatten()]  # labels以图像返回的下标来排列，从而使其聚类化
dst2 = res.reshape((img.shape))  # numpy.reshape(a, newshape, order=‘C’)   ,重塑成2维图像，h*w，3通道的数据

centers4 = np.uint8(centers4)
res = centers4[labels4.flatten()]
dst4 = res.reshape((img.shape))

centers8 = np.uint8(centers8)
res = centers8[labels8.flatten()]
dst8 = res.reshape((img.shape))

centers16 = np.uint8(centers16)
res = centers16[labels16.flatten()]
dst16 = res.reshape((img.shape))

centers64 = np.uint8(centers64)
res = centers64[labels64.flatten()]
dst64 = res.reshape((img.shape))

# 图像转换为RGB显示
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)
dst4 = cv2.cvtColor(dst4, cv2.COLOR_BGR2RGB)
dst8 = cv2.cvtColor(dst8, cv2.COLOR_BGR2RGB)
dst16 = cv2.cvtColor(dst16, cv2.COLOR_BGR2RGB)
dst64 = cv2.cvtColor(dst64, cv2.COLOR_BGR2RGB)

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

# 显示图像
title = [u'原始图像', u'聚类图像 k=2', u'聚类图像 k=4',
         u'聚类图像 k=8', u'聚类图像 k=16', u'聚类图像 k=64']
img = [img, dst2, dst4, dst8, dst16, dst64]
for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(img[i], "gray"),
    plt.title(title[i])
    plt.xticks([]), plt.yticks([])
plt.show()
