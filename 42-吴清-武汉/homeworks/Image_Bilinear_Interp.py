# -*- coding:utf-8 -*-
# 八斗AI课作业.
# 图像处理-图像邻近插值和双线性插值
# 42-吴清-武汉
# 2022-05-08
import tools.ImageHelper as tools
import cv2 as cv
import numpy as np

def do():
    # 使用邻近插值法进行向上和向下插值
    img_adjacent_up=tools.image_adjacent("lenna.png",800,800) #向上插值
    img_adjacent_down = tools.image_adjacent("lenna.png", 100, 100)  # 向下插值
    cv.imshow("img-adjacent-up",img_adjacent_up)
    cv.imshow("img-adjacent-down", img_adjacent_down)
    # 使用双线性插值进行向上和向下插值
    img_gray=tools.image_gray("lenna.png")
    img_linear_up = tools.image_linear("lenna.png", 800, 800)  # 向上插值
    img_linear_down = tools.image_linear("lenna.png", 100, 100)  # 向下插值
    cv.imshow("img-linear-up", img_linear_up)
    cv.imshow("img-linear-down", img_linear_down)
    #直方图均衡化
    image_histogram=tools.image_histogram("lenna.png");
    cv.imshow("multi-image",np.hstack([img_gray,image_histogram]))
    cv.waitKey()
    cv.destroyAllWindows()