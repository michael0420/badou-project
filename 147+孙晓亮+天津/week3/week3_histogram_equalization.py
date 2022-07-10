import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


# 灰度图均衡化
def gray_image_equal():
    img = cv.imread('snow.jpg')
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_hist_equal = cv.equalizeHist(img_gray)
    img_hist = cv.calcHist([img_hist_equal], [0], None, [256], [0, 256])

    # plt.figure()
    # plt.hist(img_hist_equal.ravel(), 256)
    # plt.show()

    # plt.figure()
    # plt.plot(img_hist)
    # plt.show()

    cv.imshow("Histogram Equalization", np.hstack([img_gray, img_hist_equal]))
    cv.waitKey(0)


# 彩色图均衡化
def color_image_equal():
    img = cv.imread('snow.jpg')
    (b, g, r) = cv.split(img)
    bh = cv.equalizeHist(b)
    gh = cv.equalizeHist(g)
    rh = cv.equalizeHist(r)
    result = cv.merge((bh, gh, rh))
    cv.imshow("dst_rgb", np.hstack([img, result]))
    cv.waitKey(0)


if __name__ == '__main__':
    gray_image_equal()
    # color_image_equal()