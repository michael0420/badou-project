import cv2
import numpy as np
from matplotlib import pyplot as plt

image='lenna.png'
# 彩色图像直方图均衡化
def colortest(image):
    img = cv2.imread(image, 1)

    # 彩色图像均衡化,需要分解通道 对每一个通道均衡化
    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    # 合并每一个通道
    result = cv2.merge((bH, gH, rH))

    cv2.imshow("dst_rgb", np.hstack([img, result]))
    cv2.waitKey(0)


#灰度图
def graytest(image):
    img = cv2.imread(image, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #计算直方图
    dst = cv2.equalizeHist(gray)
    # 直方图均衡化
    hist = cv2.calcHist([dst],[0],None,[256],[0,256])
    cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
    #绘制直方图
    plt.figure()
    plt.hist(dst.ravel(), 256)#dst。ravel()将多维数组展平为一维数组
    plt.show()
    cv2.waitKey(0)



if __name__ =='__main__':
    colortest(image)
    graytest(image)
