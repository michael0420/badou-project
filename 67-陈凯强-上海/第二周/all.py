
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("lenna.png")
#最邻近
def nearst(img,dst_width,dst_height):
    height,width,channels =img.shape
    finalImage=np.zeros((dst_height,dst_width,channels),np.uint8)
    sh=dst_height/height
    sw=dst_width/width
    print(height,width)
    print(sh,sw)
    for i in range(dst_height):
        for j in range(dst_width):
            x=round(i/sh)
            y=round(j/sw)
            print(x,y)
            finalImage[i,j]=img[x,y]
    return finalImage
#双线性
def bilinear_interpolation(img, out_width, out_height):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_height, out_width
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)
    # 如果输入的大小相等直接做拷贝
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, channel), dtype=np.uint8)
    # 比例值
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 0.5使几何中心相同
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5
                # 向下取整
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    return dst_img
#直方图
def his():
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("image_gray", gray)

    # 灰度图像的直方图，方法一
    plt.figure()
    plt.hist(gray.ravel(), 256)
    plt.show()

    '''
    # 灰度图像的直方图, 方法二
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    plt.figure()#新建一个图像
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")#X轴标签
    plt.ylabel("# of Pixels")#Y轴标签
    plt.plot(hist)
    plt.xlim([0,256])#设置x坐标轴范围
    plt.show()
    '''

    '''
    #彩色图像直方图

    image = cv2.imread("lenna.png")
    cv2.imshow("Original",image)
    #cv2.waitKey(0)

    chans = cv2.split(image)
    colors = ("b","g","r")
    plt.figure()
    plt.title("Flattened Color Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")

    for (chan,color) in zip(chans,colors):
        hist = cv2.calcHist([chan],[0],None,[256],[0,256])
        plt.plot(hist,color = color)
        plt.xlim([0,256])
    plt.show()
    '''
def equ_his():
    '''
    equalizeHist—直方图均衡化
    函数原型： equalizeHist(src, dst=None)
    src：图像矩阵(单通道图像)
    dst：默认即可
    '''

    # 获取灰度图像
    img = cv2.imread("lenna.png", 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("image_gray", gray)

    # 灰度图像直方图均衡化
    dst = cv2.equalizeHist(gray)

    # 直方图
    hist = cv2.calcHist([dst], [0], None, [256], [0, 256])

    plt.figure()
    plt.hist(dst.ravel(), 256)
    plt.show()

    cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
    cv2.waitKey(0)

    '''
    # 彩色图像直方图均衡化
    img = cv2.imread("lenna.png", 1)
    cv2.imshow("src", img)

    # 彩色图像均衡化,需要分解通道 对每一个通道均衡化
    (b, g, r) = cv2.split(img)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    # 合并每一个通道
    result = cv2.merge((bH, gH, rH))
    cv2.imshow("dst_rgb", result)

    cv2.waitKey(0)
    '''


if __name__ == '__main__':

    inputs = int(input("最邻近插值输入1\n双线性插值输入2\n直方图输入3\n直方图均衡化输入4"))
    if inputs == 1:
        inputs = input("请输入放大或缩小后的宽高 空格隔开")
        dstimg=nearst(img,int(inputs.split(' ')[0]),int(inputs.split(' ')[1]))
        # print(zoom)
        print(dstimg.shape)
        cv2.imshow("nearest interp",dstimg)
        cv2.imshow("image",img)
        cv2.waitKey(0)
    if inputs == 2:
        inputs = input("请输入放大或缩小后的宽高 空格隔开")
        img = cv2.imread('lenna.png')
        dst = bilinear_interpolation(img, int(inputs.split(' ')[0]), int(inputs.split(' ')[1]))
        cv2.imshow('bilinear interp', dst)
        cv2.imshow('img', img)
        cv2.waitKey()
    if inputs == 3:
        his()
    if inputs == 4:
        equ_his()

