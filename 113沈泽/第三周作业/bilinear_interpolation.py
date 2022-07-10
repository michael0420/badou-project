import cv2
import numpy as np

def bilinear_interpolation(img, h, w):
    src_h, src_w, channels =img.shape
    if h == src_h and w == src_w:
        return img.copy()
    dst_img =np.zeros((h, w, 3), np.uint8)
    scale_x, scale_y = float(src_w)/w, float(src_h)/h
    for i in range(3):
        for y in range(h):
            for x in range(w):
                src_x = (x + 0.5) * scale_x - 0.5
                src_y = (y + 0.5) * scale_y - 0.5

                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[y, x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img

img = cv2.imread("lenna.png")
dst = bilinear_interpolation(img, 700, 700)
cv2.imshow("bilinear_interpolation", dst)
cv2.imshow("img", img)
cv2.waitKey()
