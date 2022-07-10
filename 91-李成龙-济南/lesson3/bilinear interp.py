# coding :utf-8
"""
Task:双线性插值
Author:91-李成龙-济南
Date:2022-05-13
"""
import cv2
import numpy as np


def bilinear_interp(img, dst_h, dst_w):
    src_h, src_w, channel = img.shape
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, des_w = ", dst_h, dst_w)
    if src_h == dst_h  and src_w == dst_w:
        return img.copy()  # 若尺寸一致，则无后续操作
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)  # 定义目标图像空间大小
    sh, sw = float(src_h) / dst_h, float(src_w) / dst_w   # 计算缩放比
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                src_x = (dst_x + 0.5) * sw - 0.5     # 几何中心重合方法确定dst与src的坐标对应关系
                src_y = (dst_y + 0.5) * sh - 0.5

                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w -1 )   # 之所以选择取最小，是判断所取点的坐标是否超出边界值
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h -1 )
                temp0 = (src_x1-src_x) * img[src_y0, src_x0, i] +(src_x- src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1-src_x) * img[src_y1, src_x0, i] +(src_x- src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img


if __name__ == "__main__":
    img = cv2.imread("lenna.png")
    dst = bilinear_interp(img, 800, 800)
    cv2.imshow("bilinear_interp", dst)
    cv2.waitKey()
    cv2.destroyAllWindows()




