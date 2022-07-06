
import numpy as np
import cv2

def bilinear_inter(img, odim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = odim[1], odim[0]
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w: #尺寸与原图一样，不需做处理
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, channel), dtype=np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h #计算 原图与目标图 宽，高 比例
    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                #原图与目标图几何中心重合点
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5


                #得到几何重叠后， 判断新点是否超出原图 ，取“小”值避免超出原图点
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                #根据双线插值公式，得到 新点的 值，按 通道 分别计算
                #先计算X方向插值
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                #计算Y方向插值，并赋值
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img



img = cv2.imread('w2.png')
dst = bilinear_inter(img, (int(img.shape[1]*1.5), int(img.shape[0]*1.5)))
cv2.imshow('bilinear interp', dst)
cv2.waitKey()
