import cv2
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread('lenna.png', 0)
# print(img)
rows, cols = img.shape
img_new = img.reshape((rows * cols, 1))
img_new = np.float32(img_new)

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
compactness, labels, centers = cv2.kmeans(img_new, 4, None, criteria, 10, flag)
# print(compactness)
# print(labels)
print(centers)

dst = labels.reshape((img.shape))
# print(dst)

#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

titles = [u'原始图像', u'聚类图像']  
images = [img, dst]  

for i in range(2):  
   plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray'), 
   plt.title(titles[i])  
   plt.xticks([]),plt.yticks([])  
plt.show()