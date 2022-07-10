# -*- coding:utf-8 -*-
# 八斗AI课作业.
# 图像处理-灰度化和二值化
# 42-吴清-武汉
# 2022-4-17

import tools.ImageHelper as tools
import cv2 as cv


def do():
    img_gray=tools.image_gray("lenna.png")
    img_twovalue= tools.image_twoValue("lenna.png")
    cv.imshow("img-gray",img_gray)
    cv.imshow("img-two-value", img_twovalue)
    cv.waitKey()
    cv.destroyAllWindows()