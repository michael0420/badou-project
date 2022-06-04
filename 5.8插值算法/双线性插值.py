# -*- coding = utf-8 -*-
# @Time : 2022/5/12 10:55
# @Author : L
# @File : 双线性插值.py
# @Software : PyCharm

import cv2 as cv
import numpy as np


def bin(src_img, out_dim):
    # 取得源图像和目标图像的宽高
    src_h, src_w, src_c = src_img.shape
    dst_h, dst_w = out_dim[0], out_dim[1]
    if src_h == dst_h and src_w == dst_w:
        return src_img.copy()
    # 创建目标图像的宽高0数组
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    # 边长比，缩放比例
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    # 计算目标像素在源图像中的位置
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                src_x = dst_x * scale_x
                src_y = dst_y * scale_y

                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                temp0 = (src_x1 - src_x) * src_img[src_y0, src_x0, i] + (src_x - src_x0) * src_img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * src_img[src_y1, src_x0, i] + (src_x - src_x0) * src_img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img


if __name__ == '__main__':
    img = cv.imread("lenna.png")

    out_img = bin(img, (200, 200))

    cv.imshow("Image", out_img)

    cv.waitKey(0)
    cv.destroyAllWindows()



