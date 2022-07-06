import numpy as np
import cv2


def bilinear_interpolation(img, out_img):
    src_H, src_W,C = img.shape
    dst_h = out_img[0]
    dst_w = out_img[1]
    if src_H == dst_h and src_W == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    scale_x, scale_y = float(src_W) / dst_w, float(src_H) / dst_h
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 几何中心重合
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_W - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_H - 1)

                # 公式
                R1 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                R2 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * R1 + (src_y - src_y0) * R2)

    return dst_img


img = cv2.imread('lenna.png')
out_img = bilinear_interpolation(img,(700,700))
# print(out_img)
cv2.imshow('bilinear inter',out_img)
cv2.waitKey(0)