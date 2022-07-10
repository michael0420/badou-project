import cv2 as cv
import numpy as np


def bilinear_interpolation(img, input_shape):
    src_h, src_w, src_c = img.shape
    dst_w, dst_h = input_shape[0], input_shape[1]
    dst_img = np.zeros((dst_h, dst_w, src_c), dtype=np.uint8)
    h_scale = float(src_h)/dst_h
    w_scale = float(src_w)/dst_w

    for i in range(src_c):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                src_x = (dst_x+0.5)*w_scale-0.5
                src_y = (dst_y+0.5)*h_scale-0.5

                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0+1, src_w-1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0+1, src_h-1)

                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    return dst_img

'''
    假设原图尺寸： m * m，变化后图的尺寸为： n * n，同时设偏移z个单位后中心点可以重合
    原图中心点：（m - 1） / 2 + z ， 变化后图中心点为（n - 1） / 2 + z 
    原图与变化后图符合比例关系为 m / n
    0.5m - 0.5 + z = m / n * （0.5n - 0.5 + z）
    （1 - m / n）z = 0.5（mn - m） / n - 0.5（m - 1）
    （n - m）/ n * z  = 0.5（mn - m - mn + n） / n = 0.5（n - m） / n
    z = 0.5
'''

img_path = 'meixi.jpg'
img = cv.imread(img_path)
img_scaled_shape = (800, 600)
new_img = bilinear_interpolation(img, input_shape=img_scaled_shape)
cv.imshow('original_image', img)
cv.imshow('new_image', new_img)
cv.waitKey(0)
cv.destroyAllWindows()