import numpy as np
from matplotlib import pyplot as plt
import cv2

img = cv2.imread('lenna.png')
# 根据另一个给定的数字3来计算剩下的维度
data = img.reshape((-1,3))
data = np.float32(data)

# criteria (type, max_iter, epsilon)
# type:
# cv2.TERM_CRITERIA_EPS ：精确度（误差）满足epsilon，则停止
# cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter，则停止
# cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER：两者结合，满足任意一个结束
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# flag: 
# cv2.KMEANS_PP_CENTERS：使用kmeans++算法的中心初始化算法，即初始中心的选择使眼色相差最大
# cv2.KMEANS_RANDOM_CENTERS：每次随机选择初始中心
flag = cv2.KMEANS_RANDOM_CENTERS

# retval, bestLabels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags, centers=None)
# compactness：紧密度，返回每个点到相应重心的距离的平方和
# labels：结果标记，每个成员被标记为分组的序号，如 0,1,2,3,4...
# centers：由聚类的中心组成的数组
# data：需要聚类数据，最好是np.float32的数据，每个特征放一列
# K:  聚类个数
# bestLabels：预设的分类标签或者None
# attempts：重复试验kmeans算法次数，将会返回最好的一次结果

# 聚为2类
compactness2, labels2, centers2 = cv2.kmeans(data, 2, None, criteria, 10, flag)

# 聚为4类
compactness4, labels4, centers4 = cv2.kmeans(data, 4, None, criteria, 10, flag)

# 聚为8类
compactness8, labels8, centers8 = cv2.kmeans(data, 4, None, criteria, 10, flag)

# 聚为16类
compactness16, labels16, centers16 = cv2.kmeans(data, 4, None, criteria, 10, flag)

# 聚为64类
compactness64, labels64, centers64 = cv2.kmeans(data, 4, None, criteria, 10, flag)

#图像转换回uint8二维类型
centers2 = np.uint8(centers2)
# print(centers2)
# print(type(centers2))
print(labels2)
print(labels2.flatten())
# print(type(labels2.flatten()))
res = centers2[labels2.flatten()]
# print(res)
dst2 = res.reshape((img.shape))
# print(dst2)

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

#图像转换为RGB显示
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
dst2 = cv2.cvtColor(dst2, cv2.COLOR_BGR2RGB)
dst4 = cv2.cvtColor(dst4, cv2.COLOR_BGR2RGB)
dst8 = cv2.cvtColor(dst8, cv2.COLOR_BGR2RGB)
dst16 = cv2.cvtColor(dst16, cv2.COLOR_BGR2RGB)
dst64 = cv2.cvtColor(dst64, cv2.COLOR_BGR2RGB)

#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

#显示图像
titles = [u'原始图像', u'聚类图像 K=2', u'聚类图像 K=4',
          u'聚类图像 K=8', u'聚类图像 K=16',  u'聚类图像 K=64']  
images = [img, dst2, dst4, dst8, dst16, dst64]  
for i in range(6):  
   plt.subplot(2,3,i+1), plt.imshow(images[i], 'gray'), 
   plt.title(titles[i])  
   plt.xticks([]),plt.yticks([])  
plt.show()