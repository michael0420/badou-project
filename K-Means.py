'''
在OpenCV中，Kmeans()函数原型如下所示：
retval, bestLabels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
    data表示聚类数据，最好是np.flloat32类型的N维点集
    K表示聚类类簇数
    bestLabels表示输出的整数数组，用于存储每个样本的聚类标签索引
    criteria表示迭代停止的模式选择，这是一个含有三个元素的元组型数。格式为（type, max_iter, epsilon）
        其中，type有如下模式：
         —–cv2.TERM_CRITERIA_EPS :精确度（误差）满足epsilon停止。
         —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
         —-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
    attempts表示重复试验kmeans算法的次数，算法返回产生的最佳结果的标签
    flags表示初始中心的选择，两种方法是cv2.KMEANS_PP_CENTERS ;和cv2.KMEANS_RANDOM_CENTERS
    使用kmeans++算法的中心初始化算法，即初始中心的选择使眼色相差最大.详细可查阅kmeans++算法。
    centers表示集群中心的输出矩阵，每个集群中心为一行数据
'''

import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('C:\\Users\\LENOVO\\Desktop\\lenna.png', 0)  # 读取灰度图像，1的时候是读取彩色图像，-1的时候是读取全通道包括alpha
print(img.shape)

# 将数据转换成1维
rows, cols = img.shape[:]
data = img.reshape((rows * cols, 1))     #转行成长乘宽那么多行，和1列数据，即把每一行列的数据都提出来了
data = np.float32(data)

#停止条件 (type,max_iter,epsilon)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) #每次迭代次数达到10次时或者epsilon精度为1.0时

#设置标签
flags = cv2.KMEANS_RANDOM_CENTERS

#K-Means聚类 聚集成4类 ，返回数据
#compactness：紧密度，返回每个点到相应重心的距离的平方和
#labels：结果标记，每个成员被标记为分组的序号，如 0,1,2,3,4...等
#centers：由聚类的中心组成的数组
compactness, labels , centers = cv2.kmeans(data,4,None,criteria,10,flags)

dst = labels.reshape(img.shape[0],img.shape[1])

#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

#显示图像
titles = [u'原始图像', u'聚类图像']
images = [img, dst]
for i in range(2):
    plt.subplot(1,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])    #不显示横纵坐标
plt.show()