# -*- coding:utf-8 -*-
# 八斗AI课作业.
# 图像处理-透视变换
# 42-吴清-武汉
# 2022-5-25

import tools.ImageHelper as tools
import cv2 as cv
import  numpy as np


def do():
    imgs = tools.image_warp_martrix("photo1.jpg")
    cv.imshow("image_warp_source", imgs[0])
    cv.imshow("image_warp_martrix", imgs[1])
    cv.waitKey()
    cv.destroyAllWindows()