import cv2
import numpy as np


def drawMatchesKnn(img_1, kp_1, img_2, kp_2, matches_):
    """
    将输入的两幅图像生成一幅合图，并用线连接匹配好的关键点
    输入：
        img_1: 待处理的图像1
        kp_1:  图像1的关键点信息，SIFT计算所得格式
        img_2: 待处理的图像2
        kp_2: 图像2的关键点信息，SIFT计算所得格式
        matches: 关键点的匹配信息
    输出:
        两幅图的合图，以及匹配点的连线
    """
    h1, w1 = img_1.shape[0:2]
    h2, w2 = img_2.shape[0:2]
    print('输入图1的形状', img_1.shape)
    print('输入图2的形状', img_2.shape)
    # 1、生成合图
    together = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)  # 必须用np.uint8指定数据类型,否则后面有错误
    together[0:h1, 0:w1] = img_1
    together[0:h2, w1:w1 + w2] = img_2
    print('合成图的形状', together.shape)
    # 2、提取对应的匹配好的点
    kpIdx1 = [kp.queryIdx for kp in matches_]  # 列表生成器，提取匹配点中第一图的点的索引，一般就是一个数字序号
    kpIdx2 = [kp.trainIdx for kp in matches_]  # 列表生成器，提取匹配点中第二图的点的索引
    kphw1 = np.int32([kp_1[k].pt for k in kpIdx1])  # 列表生成器，根据上一步的索引，第几个，到原来的点集合中提取坐标（w,h)
    kphw2 = np.int32([kp_2[k].pt for k in kpIdx2]) + (w1, 0)  # kp_1中每个点是一个对象，可以用.pt就是从点的对象中，提取出对应的行列坐标元组
    # 关于图像的相关点的坐标都是 （宽，高），所以要加（w1，0）
    # 3、画出连线
    for (x1, y1), (x2, y2) in zip(kphw1, kphw2):
        cv2.line(together, (x1, y1), (x2, y2), (0, 0, 255))  # 在原图中画线，红色
    # cv2.namedWindow("match", cv2.WINDOW_NORMAL)  # 创建一个展示窗口
    # cv2.imshow("match", together)  # 在窗口中显示图片
    return together


img1 = cv2.imread('iphone1.png')
img2 = cv2.imread('iphone2.png')

sift = cv2.xfeatures2d.SIFT_create()  # 实例化
# 关键点 key_points, 描述descriptor = sift.detectAndCompute(gray, None)  # 关键点（特征点）探测和估算
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
# 以下代码为：在原图上把特征点标识出来，flag是用小圆圈的意思
# img = cv2.drawKeypoints(image=img, keypoints=keypoints, outImage=img,
#                         color=(51, 163, 236),
#                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# img = cv2.drawKeypoints(gray1, kp1, gray1)

bf = cv2.BFMatcher(cv2.NORM_L2)  # 蛮力匹配器实例化（L1是出租车(或曼哈顿)距离(绝对值之和),L2是欧氏距离(平方和的平方根))

# 方法一, bf.match匹配
matches = bf.match(des1, des2)  # 蛮力匹配,只匹配最适合的点. 但不一定是从距离从小到大的排序，
# matches中包括一组匹配好的点，其中每个匹配的点包括：queryIdx：图1索引，trainIdx图2索引，和它们之间的距离
# for i in matches[:20]:  # 但不一定是从距离最小的排序
#     print(i.distance)
matches = sorted(matches, key=lambda x: x.distance)  # 按照距离从小到大排序
flag = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
match_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=flag)
cv2.imshow('match1', match_result)

# 方法二， bf.knnMatch匹配
match = bf.knnMatch(des1, des2, k=2)  # 蛮力匹配。输出2组匹配点（最接近，次接近），每组中包括（图1索引，图2索引，距离-越小越好）
good_match = []
for m, n in match:  # 该步计算可以去除一部分相似点
    if m.distance < 0.5*n.distance:  # 最接近的匹配距离小于次接近的匹配距离的一半，证明最接近的匹配效果越好。
        good_match.append(m)
print('len', len(good_match))
print('第一幅图点的索引：{}，第二幅点索引：{}，它们之间距离为{}'.format(good_match[2].queryIdx,
                                               good_match[2].trainIdx, good_match[2].distance))
match_result = drawMatchesKnn(img1, kp1, img2, kp2, good_match[0:20])

cv2.imshow('match2', match_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
