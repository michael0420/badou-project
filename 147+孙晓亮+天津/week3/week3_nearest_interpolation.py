import cv2 as cv
import numpy as np


def nearest_interpolation(img, input_shape):
    h, w, c = img.shape
    print(img.shape)
    h_scale = input_shape[1] / h
    w_scale = input_shape[0] / w
    img_new = np.zeros((input_shape[1], input_shape[0], c), dtype=np.uint8)
    for i in range(input_shape[1]):
        for j in range(input_shape[0]):
            if int(i/h_scale+0.5) <= h-1:
                x = int(i/h_scale+0.5)
            else:
                x = int(i/h_scale)
            if int(j/w_scale+0.5) <= w-1:
                y = int(j/w_scale+0.5)
            else:
                y = int(j/w_scale)
            # print(x, y)
            img_new[i, j] = img[x, y]
    return img_new


img_path = 'meixi.jpg'
img = cv.imread(img_path)
img_scaled_shape = (800, 600)
new_img = nearest_interpolation(img, input_shape=img_scaled_shape)
cv.imshow('original_image', img)
cv.imshow('new_image', new_img)
cv.waitKey(0)
cv.destroyAllWindows()
