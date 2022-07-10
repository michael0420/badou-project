import cv2
import numpy as np

img = cv2.imread("../image/lenna.png")


def assign(img, shape):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = shape

    if src_h == dst_h and src_w == dst_w:
        return img.copy()

    dst_img = np.zeros((dst_h, dst_w, channel))
    w_rate, h_rate = src_w / dst_w, src_h / dst_h

    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                #src_x = dst_x * w_rate
                #src_y = dst_y * h_rate
                src_x = (dst_x + 0.5) * w_rate - 0.5
                src_y = (dst_y + 0.5) * h_rate - 0.5

                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1) / 255

    return dst_img


new = assign(img, (650, 650))
cv2.imshow("bilinear interp", new)
cv2.imshow("image", img)
cv2.waitKey(0)