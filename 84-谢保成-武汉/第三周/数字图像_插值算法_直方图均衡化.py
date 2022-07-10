import cv2
import numpy as np

ImageFile = '/filehome/PythonProjects/badou-project/84-谢保成-武汉/image/lenna.png'

# ===========================双线性插值=======================
img = cv2.imread('/filehome/PythonProjects/badou-project/84-谢保成-武汉/image/lenna.png', 1)
imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1]
mode = imgInfo[2]
# 放大/缩小； 等比例缩放：非等比例缩放# 缩小一半
dstHeight = int(height * 0.5)
dstWidth = int(width * 0.5)
img = cv2.resize(img, (dstWidth, dstHeight))
imgInfo = img.shape
height = imgInfo[0]
width = imgInfo[1]
mode = imgInfo[2]
cv2.imshow('img', img)
# 最临近域插值，双线性插值，像素关系重采样，立方插值
dst = cv2.resize(img, (dstWidth, dstHeight))
# cv2.imshow('image', dst)
'''双线性插值算法原理实现'''
dstImage = np.zeros((dstHeight, dstWidth, 3), np.uint8)
for i in range(0, dstHeight):
    for j in range(0, dstWidth):
        iNew = int(i * (height * 1.0 / dstHeight))
        jNew = int(j * (width * 1.0 / dstWidth))
        dstImage[i, j] = img[iNew, jNew]
cv2.imshow('dst', dstImage)


# ===========================最临近域插值=======================
def nearestInterp(img):
    height, width, channels = img.shape
    emptyImage = np.zeros((800, 800, channels), np.uint8)
    sh = 800 / height
    sw = 800 / width
    for i in range(800):
        for j in range(800):
            x = int(i / sh)
            y = int(j / sw)
            emptyImage[i, j] = img[x, y]
    return emptyImage


img = cv2.imread(ImageFile)
dst = nearestInterp(img)
cv2.imshow('nearest interp', dst)

# ===========================直方图均衡化=====================================
image = cv2.imread(ImageFile)
# 灰度图片的直方图均衡化
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将彩色图片转换为灰度图
cv2.imshow('grayImg', gray)
result = cv2.equalizeHist(gray)
cv2.imshow('grayHist', result)

# 彩色图片的直方图均衡化
colorImg = cv2.imread(ImageFile)
cv2.imshow('colorImg', colorImg)
(b, g, r) = cv2.split(colorImg)  # 分解通道
# 对各个通道进行直方图均衡化
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 将均衡化之后的数据合并在一起
result = cv2.merge((bH, gH, rH))
cv2.imshow('colorHist', result)

# YUV图片的直方图均衡化
colorImg = cv2.imread(ImageFile)
# 将彩色图片转换为yuv图
YUVImg = cv2.cvtColor(colorImg, cv2.COLOR_BGR2YUV)
channelYUV = cv2.split(YUVImg)
channelYUV[0] = cv2.equalizeHist(channelYUV[0])
channels = cv2.merge(channelYUV)
result = cv2.cvtColor(channels, cv2.COLOR_YUV2BGR)
cv2.imshow('YUVHist', result)

cv2.waitKey(0)

