import cv2

#设置高低阈值
lowThreshold = 0
max_lowThreshold = 100
ratio = 3
kernel_size = 3

img = cv2.imread('lenna.png')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def Canny(lowThreshold,ratio,img_gray):
    detected_edges = cv2.GaussianBlur(img_gray,(3,3),0) #高斯滤波处理
    detected_edges = cv2.Canny(detected_edges,
                               lowThreshold,
                               lowThreshold * ratio,
                               apertureSize=kernel_size)  # 边缘检测
    cv2.imshow('canny demo', detected_edges)
    cv2.waitKey()

Canny(0,ratio,img_gray)