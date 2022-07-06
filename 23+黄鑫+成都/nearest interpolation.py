import cv2
import numpy as np


def function(img, h, w):  # define three variable
    height, width, channels = img.shape  # 读取图片的高，宽，通道
    emptyImage = np.zeros((h, w, channels), np.uint8)  # 创建一张高h，宽w通道为，通道为channels，数据类型为uint8的空白图片
    sh = h / height  # sh缩放的高
    sw = w / width  # sw缩放的宽
    for x in range(h):
        for y in range(w):  # 遍历创建的图片
            i = int(x / sh + 0.5)
            j = int(y / sw + 0.5)
            emptyImage[x, y] = img[i, j]  # 用最邻近插值法找到创建的图片对应的原图的像素点
    return emptyImage


img = cv2.imread("C:\\Users\\LENOVO\\Desktop\\lenna.png")  # 读取目标图片BGR
h = int(input("请输入高:"))
w = int(input("请输入宽:"))
zoom = function(img, h, w)
print(zoom)
cv2.imshow("nearest interpolation", zoom)  # show创建的图片
cv2.imshow("image", img)  # 展示原图
cv2.waitKey(0)
