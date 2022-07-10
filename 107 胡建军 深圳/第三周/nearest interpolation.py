import cv2
import numpy as np


def read_img(path):
    img = cv2.imread(path)
    return img


def nearest_interp(img, m, n):
    h, w , c = img.shape
    new_img = np.zeros([m, n, c], np.uint8)
    kh = h/m
    kw = w/n
    for i in range(m):
        for j in range(n):
            y = min(round(i * kh), h-1)#注意边界溢出
            x = min(round(j * kw), w-1)
            new_img[i, j] = img[y, x]
    return new_img

if __name__ == '__main__':
    path = './lenna.png'
    img = read_img(path)
    m = 800
    n = 1200
    resize_img = nearest_interp(img, m, n)
    cv2.imshow('nearst interpolation', resize_img)
    cv2.imshow('original image', img)
    cv2.waitKey(0)


