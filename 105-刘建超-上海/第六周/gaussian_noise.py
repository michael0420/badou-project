#!/usr/bin/python
# encoding=utf-8

import cv2
import numpy as np
import random

'''高斯噪声'''


def GaussianNoise(src, means, sigma, percetage):
    NoiseImg = src
    NoiseNum = int(src.shape[0] * src.shape[1] * percetage)
    for i in range(NoiseNum):   #每次取一个随机点
        randR = random.randint(0, src.shape[0] - 1)  # 随机生成行
        randC = random.randint(0, src.shape[1] - 1)  # 随机生成列
        NoiseImg[randR, randC] = NoiseImg[randR, randC] + random.gauss(means, sigma)    #在原有像素灰度值上加上随机数
        if NoiseImg[randR, randC] < 0:
            NoiseImg[randR, randC] = 0
        elif NoiseImg[randR, randC] > 255:
            NoiseImg[randR, randC] = 255
    return NoiseImg


img = cv2.imread("lenna.png", 0)
img_result = GaussianNoise(img, 4, 8, 0.8)
img = cv2.imread("lenna.png", 1)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("origin", img_gray)
cv2.imshow("result", img_result)
cv2.waitKey(0)
cv2.destroyAllWindows()