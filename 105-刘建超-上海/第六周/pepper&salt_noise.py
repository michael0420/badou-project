#!/usr/bin/env python
# coding=utf-8

import cv2
import numpy as np
import random

'''椒盐噪声'''


def PepperSaltNoise(src, percetage):
    NoiseImg = src
    NoiseNum = int(src.shape[0] * src.shape[1] * percetage)
    for i in range(NoiseNum):   #每次取一个随机点
        randR = random.randint(0, src.shape[0] - 1)  # 随机生成行
        randC = random.randint(0, src.shape[1] - 1)  # 随机生成列
        if random.random() <= 0.5:
            NoiseImg[randR, randC] = 0
        else:
            NoiseImg[randR, randC] = 255
    return NoiseImg


img = cv2.imread("lenna.png")
img_result = PepperSaltNoise(img, 0.1)
img = cv2.imread("lenna.png", 1)
cv2.imshow("origin", img)
cv2.imshow("result", img_result)
cv2.waitKey()
cv2.destroyAllWindows()