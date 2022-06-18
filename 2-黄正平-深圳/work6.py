"""
图像匹配——SIFT点特征匹配实现步骤：
    （1）读取图像；
    （2）定义sift算子；
    （3）通过sift算子对需要匹配的图像进行特征点获取；
        a.可获取各匹配图像经过sift算子的特征点数目
    （4）可视化特征点（在原图中标记为圆圈）；
        a.为方便观察，可将匹配图像横向拼接
    （5）图像匹配（特征点匹配）；
        a.通过调整ratio获取需要进行图像匹配的特征点数量（ratio值越大，匹配的线条越密集，但错误匹配点也会增多）
        b.通过索引ratio选择固定的特征点进行图像匹配
    （6）将待匹配图像通过旋转、变换等方式将其与目标图像对齐
"""

import cv2              # opencv版本需为3.4.2.16
import numpy as np      # 矩阵运算库
import time             # 时间库
from PIL import Image
# 一、实现sift
img = Image.open('lena.jpeg')
img2 = img.rotate(45)  # 逆时针旋转45°
img2.save("lena_rot45.jpeg")
img2.show()

original_lena = cv2.imread('lena.jpeg')          # 读取lena原图
lena_rot45 = cv2.imread('lena_rot45.jpeg')       # 读取lena旋转45°图

sift = cv2.xfeatures2d.SIFT_create()

# 获取各个图像的特征点及sift特征向量
# 返回值kp包含sift特征的方向、位置、大小等信息；des的shape为（sift_num， 128）， sift_num表示图像检测到的sift特征数量
(kp1, des1) = sift.detectAndCompute(original_lena, None)
(kp2, des2) = sift.detectAndCompute(lena_rot45, None)

# 特征点数目显示
print("=========================================")
print("=========================================")
print('lena 原图  特征点数目：', des1.shape[0])
print('lena 旋转图 特征点数目：', des2.shape[0])
print("=========================================")
print("=========================================")

# 举例说明kp中的参数信息
for i in range(2):
    print("关键点", i)
    print("数据类型:", type(kp1[i]))
    print("关键点坐标:", kp1[i].pt)
    print("邻域直径:", kp1[i].size)
    print("方向:", kp1[i].angle)
    print("所在的图像金字塔的组:", kp1[i].octave)

print("=========================================")
print("=========================================")
"""
首先对原图和旋转图进行特征匹配，即图original_lena和图lena_rot45
"""
# 绘制特征点，并显示为红色圆圈
sift_original_lena = cv2.drawKeypoints(original_lena, kp1, original_lena, color=(255, 0, 255))
sift_lena_rot45 = cv2.drawKeypoints(lena_rot45, kp2, lena_rot45, color=(255, 0, 255))

sift_cat1 = np.hstack((sift_original_lena, sift_lena_rot45))        # 对提取特征点后的图像进行横向拼接
cv2.imwrite("sift_cat1.png", sift_cat1)
print('原图与旋转图 特征点绘制图像已保存')
cv2.imshow("sift_point1", sift_cat1)
cv2.waitKey()

# 特征点匹配
# K近邻算法求取在空间中距离最近的K个数据点，并将这些数据点归为一类
start = time.time()     # 计算匹配点匹配时间
bf = cv2.BFMatcher()
matches1 = bf.knnMatch(des1, des2, k=2)
print('用于 原图和旋转图 图像匹配的所有特征点数目：', len(matches1))

# 调整ratio
# ratio=0.4：对于准确度要求高的匹配；
# ratio=0.6：对于匹配点数目要求比较多的匹配；
# ratio=0.5：一般情况下。
ratio1 = 0.5
good1 = []

for m1, n1 in matches1:
    # 如果最接近和次接近的比值大于一个既定的值，那么我们保留这个最接近的值，认为它和其匹配的点为good_match
    if m1.distance < ratio1 * n1.distance:
        good1.append([m1])

end = time.time()
print("匹配点匹配运行时间:%.4f秒" % (end-start))

# 通过对good值进行索引，可以指定固定数目的特征点进行匹配，如good[:20]表示对前20个特征点进行匹配
match_result1 = cv2.drawMatchesKnn(original_lena, kp1, lena_rot45, kp2, good1, None, flags=2)
cv2.imwrite("match_result1.png", match_result1)

print('原图与旋转图 特征点匹配图像已保存')
print("=========================================")
print("=========================================")
print("原图与旋转图匹配对的数目:", len(good1))

for i in range(2):
    print("匹配", i)
    print("数据类型:", type(good1[i][0]))
    print("描述符之间的距离:", good1[i][0].distance)
    print("查询图像中描述符的索引:", good1[i][0].queryIdx)
    print("目标图像中描述符的索引:", good1[i][0].trainIdx)

print("=========================================")
print("=========================================")
cv2.imshow("original_lena and lena_rot45 feature matching result", match_result1)
cv2.waitKey()

# 将待匹配图像通过旋转、变换等方式将其与目标图像对齐，这里使用单应性矩阵。
# 单应性矩阵有八个参数，如果要解这八个参数的话，需要八个方程，由于每一个对应的像素点可以产生2个方程(x一个，y一个)，那么总共只需要四个像素点就能解出这个单应性矩阵。
if len(good1) > 4:
    ptsA = np.float32([kp1[m[0].queryIdx].pt for m in good1]).reshape(-1, 1, 2)
    ptsB = np.float32([kp2[m[0].trainIdx].pt for m in good1]).reshape(-1, 1, 2)
    ransacReprojThreshold = 4
    # RANSAC算法选择其中最优的四个点
    H, status =cv2.findHomography(ptsA, ptsB, cv2.RANSAC, ransacReprojThreshold)
    imgout = cv2.warpPerspective(lena_rot45, H, (original_lena.shape[1], original_lena.shape[0]),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    cv2.imwrite("imgout.png", imgout)
    cv2.imshow("lena_rot45's result after transformation", imgout)
    cv2.waitKey()


# 二、实现ransac
from typing import Sized
import numpy as np
import matplotlib.pyplot as plt
import random
import math

SIZE = 50
a = 2
b = 3

X = np.linspace(0, 10, SIZE)
Y = a * X + b

random_x = []
random_y = []

for i in range(SIZE):
    random_x.append(X[i] + random.uniform(-0.5, 0.5))
    random_y.append(Y[i] + random.uniform(-0.3, 0.3))

for i in range(SIZE):
    random_x.append(random.uniform(0, 10))
    random_y.append(random.uniform(3, 23))

fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)

# 使用ransac迭代
iter_num = 1000
eplison = 0.25

best_a = 0  # x
best_b = 0  # b
best_inliers_num = 0  # 内点数目

p_our = 0.9  # 自定义的概率，k次迭代中至少有一次全是内点的概率
# for i in range(iter_num):
i = 0
while (i < iter_num):
    print("--- iter_num:{}/{}---".format(i, iter_num))
    # 1 得到模型估计点
    indx1, indx2 = random.sample(range(len(random_x)), 2)
    point1_x = random_x[indx1]
    point1_y = random_y[indx1]
    point2_x = random_x[indx2]
    point2_y = random_y[indx2]

    # 2 得到模型参数
    a = (point2_y - point1_y) / (point2_x - point1_x)
    b = point2_y - a * point2_x

    # 3 判断模型成绩（统计内点数量）
    cur_inliner_num = 0  # 当前模型的内点数量
    err = 0  # 统计每一次的平均误差
    for j in range(len(random_x)):

        # 计算距离
        cur_point_x = random_x[j]
        cur_point_y = random_y[j]
        cur_dis = abs(a * cur_point_x - cur_point_y + b) / math.sqrt(a * a + 1)
        err += cur_dis * cur_dis
        if (cur_dis <= eplison):
            cur_inliner_num += 1
    err /= len(random_x)
    print("iter: {}, cur_inliner_num:{}, best_inliner_num:{}, err:{}".format(i, cur_inliner_num, best_inliers_num, err))

    i += 1

    # 如果当前模型的内点数量大于最好的模型的内点数量，更新a,b
    if (cur_inliner_num > best_inliers_num):
        P = cur_inliner_num / (2 * SIZE)  # 计算内点概率P（假设开始不知道）
        best_inliers_num = cur_inliner_num  # 更新最优内点的数量
        best_a = a  # 更新模型参数
        best_b = b
        iter_num = math.log(1 - p_our) / math.log(1 - pow(P, 2))  # 更新迭代次数
        i = 0  # 当前迭代置为0
        print("[update] iter:{}, a:{}, b:{}, P:{}, best_inliner_num:{}, iter_num:{}".format(i, best_a, best_b, P,
                                                                                            best_inliers_num, iter_num))

    if (best_inliers_num > SIZE - 10):
        print("iter[{}] converge")
        break

    ax1.cla()
    ax1.set_title("Ransac")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    # 设置x,y轴的范围 0-10，0-25
    plt.xlim((0, 10))
    plt.ylim((0, 25))
    ax1.scatter(random_x, random_y, color='b')

    Y_pred = best_a * np.asarray(random_x) + best_b
    ax1.plot(random_x, Y_pred, color='r')

    text = "best_a = " + str(best_a) + "\nbest_b = " + str(best_b)
    plt.text(5, 10, text, fontdict={'size': 8, 'color': 'r'})
    plt.pause(0.1)


# 三、实现均值/差值hash
# 1、均值hash
def aHash(img):
    #缩放为8*8
    img=cv2.resize(img,(8,8),interpolation=cv2.INTER_CUBIC)
    #转换为灰度图
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #s为像素和初值为0，hash_str为hash值初值为''
    s=0
    hash_str=''
    #遍历累加求像素和
    for i in range(8):
        for j in range(8):
            s=s+gray[i,j]
            #求平均灰度
    avg=s/64
    #灰度大于平均值为1相反为0生成图片的hash值
    for i in range(8):
        for j in range(8):
            if gray[i,j]>avg:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'
    return hash_str


# 2、差值hash

def dHash(img):
#缩放8*8
    img=cv2.resize(img,(9,8),interpolation=cv2.INTER_CUBIC)
#转换灰度图
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hash_str=''
#每行前一个像素大于后一个像素为1，相反为0，生成哈希
    for i in range(8):
        for j in range(8):
            if gray[i,j]>gray[i,j+1]:
                hash_str=hash_str+'1'
            else:
                hash_str=hash_str+'0'

