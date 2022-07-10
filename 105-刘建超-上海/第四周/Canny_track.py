#!/usr/bin/python
# encoding=gbk

import cv2

'''
cv2.createTrackbar()    #设置调节杠
共有5个参数，其实这五个参数看变量名就大概能知道是什么意思了
第一个参数，是这个trackbar对象的名字
第二个参数，是这个trackbar对象所在面板的名字
第三个参数，是这个trackbar的默认值,也是调节的对象
第四个参数，是这个trackbar上调节的范围(0~count)
第五个参数，是调节trackbar时调用的回调函数名
'''


def CannyThreshold(lowThreshold):
    detected_edges = cv2.GaussianBlur(img_gray, (3, 3), 0)  # 高斯滤波
    img_canny = cv2.Canny(detected_edges, lowThreshold, lowThreshold * ratio, apertureSize=kernel_size)  # 边缘检测
    dst = cv2.bitwise_and(img, img, mask=img_canny)  # 用原始颜色添加到检测的边缘上
    cv2.imshow("canny demo", dst)


lowThreshold = 0
min_maxThreshold = 100
ratio = 3
kernel_size = 3

img = cv2.imread("lenna.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.namedWindow("canny demo")
cv2.createTrackbar('Min threshold', "canny demo", lowThreshold, min_maxThreshold, CannyThreshold)
CannyThreshold(lowThreshold)  # 初始化
if cv2.waitKey(0) == 27:  # wait for ESC key to exit cv2
    cv2.destroyAllWindows()
