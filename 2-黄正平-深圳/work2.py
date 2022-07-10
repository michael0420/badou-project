# 第二次作业
import cv2
import numpy as np


# 1、最近邻插值算法
# 输入图像高宽，使用最近邻插值实现图像resize,可以任意设置目标图像的高宽
def NearestInterpolation(srcImg, dstH, dstW):
    srcH, srcW, _ = srcImg.shape
    srcImg = np.pad(srcImg, ((0, 1), (0, 1), (0, 0)), mode='reflect')
    dstImg = np.zeros((dstH, dstW, 3), dtype=np.uint8)
    for dstX in range(dstH):
        for dstY in range(dstW):
            x = dstX * (srcH/dstH)
            y = dstY * (srcW/dstW)
            srcX = int(x)
            srcY = int(y)
            u = x - srcX
            v = y - srcY
            if u > 0.5:
                srcX += 1
            if v > 0.5:
                srcY += 1
            dstImg[dstX, dstY] = srcImg[srcX, srcY]
    return dstImg.astype(np.uint8)


# 2、双线性插值算法
def BiLinear(srcImg, dstH, dstW):
    srcH, srcW, _ = srcImg.shape
    srcImg = np.pad(srcImg, ((0, 1), (0, 1), (0, 0)), mode='reflect')
    dstImg = np.zeros((dstH, dstW, 3), dtype=np.uint8)
    for dstX in range(dstH):
        for dstY in range(dstW):
            x = dstX * (srcH/dstH)
            y = dstY * (srcW/dstW)
            srcX = int(x)
            srcY = int(y)
            u = x - srcX
            v = y - srcY
            dstImg[dstX, dstY] = (1 - u) * (1 - v) * srcImg[srcX, srcY] + u * (1 - v) * srcImg[srcX + 1, srcY] + \
                                 (1 - u) * v * srcImg[srcX, srcY + 1] + u * v * srcImg[srcX + 1, srcY + 1]
    return dstImg.astype(np.uint8)


# 3、直方图均衡化
img = cv2.imread(r"E:\八斗作业\badou\作业\Lenna.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

dst = cv2.equalizeHist(gray)

hist = cv2.calcHist([dst], [0], None, [256], [0, 256])

plt.figure()
plt.hist(dst.ravel(), 256)
plt.show()

cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
cv2.waitKey(0)



# 彩色图像直方图均衡化
img = cv2.imread(r"Lenna.jpg", 1)
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