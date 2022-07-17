# -*- coding:utf-8 -*-
"""
作者：YSen
日期：2022年07月16日
功能：双线性插值
"""

import cv2
import numpy as np


def cv_show(name, img):
    """
    图像的显示，创建一个窗口
    利用opencv库的一些指令显示图像，集成成为一个函数
    :param name: 创建的窗口的名字,是一个字符串类型；
    :param img:传入的需要显示的图像名字，是一个变量名
    :return:None
    """
    cv2.imshow(name, img)
    cv2.waitKey(0)  # 等待时间，毫秒级的等待时间，0表示当按下任意键时终止窗口显示
    cv2.destroyAllWindows()  # 进行触发关闭窗口


def bilinear_interpolation(img, out_dim):
    src_h, src_w, channel = img.shape  # 原图片的高、宽、通道数
    dst_h, dst_w = out_dim[1], out_dim[0]  # 输出图片的高、宽
    print('src_h,src_w=', src_h, src_w)
    print('dst_h,dst_w=', dst_h, dst_w)

    if src_h == dst_h and src_w == dst_w:
        return img.copy()

    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h  # 缩放比例
    for i in range(3):  # 指定通道数，对channel循环
        for dst_y in range(dst_h):  # 指定 高，对height循环
            for dst_x in range(dst_w):  # 指定 宽，对width循环

                # 源图像和目标图像几何中心的对齐
                # src_x = (dst_x + 0.5) * srcWidth/dstWidth - 0.5
                # src_y = (dst_y + 0.5) * srcHeight/dstHeight - 0.5
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                # 计算在源图上四个近邻点的位置
                src_x0 = int(np.floor(src_x))
                src_y0 = int(np.floor(src_y))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y1 = min(src_y0 + 1, src_h - 1)

                # 双线性插值
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img


img = cv2.imread('../data/cat.jpg')
print("原图像的形状:{}".format(img.shape))
src_h, src_w, channel = img.shape  # 原图片的高、宽、通道数

bst_h = 2 * src_h
bst_w = 2 * src_w

dst = bilinear_interpolation(img, (bst_w, bst_h))
print("插值后的图像形状:{}".format(dst.shape))
cv_show("blinear", dst)
