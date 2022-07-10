import cv2
import numpy as np


def bilinear_interpolation(img, out_dim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim[0], out_dim[1]
    if src_h == dst_h and src_w == dst_w:
        return img.copy()  # 当source picture 与destination picture equal height&width时 返回一个复制的图像
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h  # 缩放比例
    dst_img = np.zeros((dst_h, dst_w, 3), np.uint8)  # 否则创建一个3通道的大小与目标图相同的图片，数据类型为np.uint8
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # find origin x &y coordinate of dst x and y
                # use geometric center of symmetry

                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                # find the coordinate points of which will be used to compute the interpolation
                # incase for the points out the threshold

                src_x0 = int(np.floor(src_x))  # 向下取整
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # calculate the interpolation
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img


if __name__ == '__main__':
    img = cv2.imread("C:\\Users\\LENOVO\\Desktop\\lenna.png")
    dst = bilinear_interpolation(img, (600, 600))
    cv2.imshow('bilinear interp', dst)
    cv2.waitKey()
