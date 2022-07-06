import numpy as np
import cv2
import random


def gaussian_noise(img_in, mean, sigma, percent):
    """
    将输入的图像增加高斯噪声
    输入：
        img_in:待处理的数字图像-灰度图，
        mean:高斯函数的均值、
        sigma:高斯函数的方差、
        percent:增加噪声的比例
    输出：增加高斯噪声后的数字图像- 灰度图
    """
    row, col = img_in.shape
    number = int(row * col * percent)  # 后面只有整数才能for循环
    img_out = img_in.copy()
    for i in range(number):
        randX = random.randint(0, row - 1)  # 生成随机坐标
        randY = random.randint(0, col - 1)
        img_out[randX, randY] = img_in[randX, randY] + random.gauss(mean, sigma)  # 增加噪声
        if img_out[randX, randY] < 0:
            img_out[randX, randY] = 0
        if img_out[randX, randY] > 255:
            img_out[randX, randY] = 255
    return img_out


def pepper_salt_noise(img_in, percent):
    """
    将输入的图像增加椒盐噪声
    输入：
        img_in:待处理的数字图像
        percent:增加噪声的比例
    输出：增加高斯噪声后的数字图像
    """
    row, col, channel = img_in.shape
    number = int(row * col * percent)
    img_out = img_in.copy()
    for i in range(number):
        randX = random.randint(0, row - 1)
        randY = random.randint(0, col - 1)
        if random.random() <= 0.5:  # random.random()生成一个0-1之间的中间的小数
            img_out[randX, randY] = np.array([0, 0, 0])
        else:
            img_out[randX, randY] = np.array([255, 255, 255])
    return img_out


# 作业一：聚类kmeans 层次聚类 密度聚类
'''
# Kmeans聚类 方法一 sklearn
x = 待处理数据
from sklearn.cluster import KMeans
clf = KMeans(n_clusters=3)  # 聚成3类的实例
y_pred = clf.fit_predict(x)  # 预测或者输出结果（对应类别的数组)

Kmeans聚类 方法二 openCV
retval, bestLabels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
    retval：紧密度，返回每个点到相应重心的距离的平方和
    bestLabels：用标记输出的结构，每个成员被标记为分组的序号，如 0,1,2,3,4...等
    centers ：由聚类的中心组成的数组
    data表示聚类数据，最好是np.float32类型的N维点集
    K表示聚类类簇数
    bestLabels 预设的分类标签或者None
    criteria表示迭代停止的模式选择，这是一个含有三个元素的元组型数。格式为（type, max_iter, epsilon）
        其中，type有如下模式：
         —–cv2.TERM_CRITERIA_EPS :精确度（误差）满足epsilon停止。
         —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
         —-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
    attempts表示 重复试验kmeans算法的次数，算法返回产生的最佳结果的标签
    flags表示初始中心的选择，两种方法是cv2.KMEANS_PP_CENTERS ;和cv2.KMEANS_RANDOM_CENTERS
    centers表示集群中心的输出矩阵，每个集群中心为一行数据,可以缺省
    
层次聚类
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
y = linkage(x, 'ward')  # 输出聚类过程
dn = dendrogram(y)  # 画树状图
z = fcluster(y) # 输出类似label的聚类结果

密度聚类
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.4, min_samples=9)  # 实例化
dbscan.fit(x)  # 训练数据
label_pred = dbscan.labels  # 输出聚类结果labels
'''

img = cv2.imread('lenna.png')
# cv2.imshow('image', img)
h, w, ch = img.shape
data = img.reshape((-1, 3))  # 把512*512压缩成1维，三通道不变
data = np.float32(data)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)  # 终止条件迭代次数10，eps=1
flags = cv2.KMEANS_RANDOM_CENTERS  # 初始点随机
ret, label, center = cv2.kmeans(data, 8, None, criteria, 10, flags)  # 聚成8类，聚10次挑效果最好的
center = np.uint8(center)  # 将浮点数转换成图像的0-255格式
print('kmeans聚类的结果label', label.flatten())
# center[2] = np.array([255, 0, 0])  # 换颜色验证
print('聚类后中心点的像素', center)
img_new = center[label]  # 利用索引生成新的数组
print('直接生成的图像形状', img_new.shape)
img_new = img_new.reshape((h, w, 3))
print('转变形状后等待输出的图像形状', img_new.shape)
cv2.imshow('original kmeans==8', np.hstack((img, img_new)))

'''
噪声
from skimage import util
noise = util.random_noise(img, model='gaussian')

mode： 可选择，str型，表示要添加的噪声类型
gaussian：高斯噪声
localvar：高斯分布的加性噪声，在“图像”的每个点处具有指定的局部方差。
poisson：泊松噪声
salt：盐噪声，随机将像素值变成1
pepper：椒噪声，随机将像素值变成0或-1，取决于矩阵的值是否带符号
s&p：椒盐噪声
speckle：均匀噪声
'''
# 作业二 高斯噪声
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gauss = gaussian_noise(gray, 2, 10, 0.8)
cv2.imshow('gaussian noise before and after', np.hstack((gray, img_gauss)))

# 作业三 椒盐噪声
img_pepper_salt = pepper_salt_noise(img, 0.2)
# img2 = cv2.cvtColor(img_pepper_salt, cv2.COLOR_BAYER_BG2RGB)  # cv2是BGR格式，如果用plt需要增加BGR2RGB转换
img3 = cv2.medianBlur(img_pepper_salt, 5)  # 中值滤波去除噪声
cv2.imshow('pepper salt noise before and after', np.hstack((img, img_pepper_salt, img3)))

cv2.waitKey(0)
