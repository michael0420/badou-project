import cv2 as cv
import numpy as np

image = cv.imread('meixi.jpg')
image_rgb = cv.cvtColor(image, cv.COLOR_RGB2BGR)


# 实现灰度化
def rgb_to_gray(image):
    h, w = image.shape[0], image.shape[1]
    image_gray = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            image_gray[i, j] = int(image[i, j, 0]*0.3 + image[i, j, 1]*0.59 + image[i, j, 2]*0.11)
    return image_gray


def gray_to_binary(image):
    h, w = image.shape[0], image.shape[1]
    image_binary = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if image[i, j]/255.0 > 0.5:
                image_binary[i, j] = 255  # 为了方便显示，将1改为255.
            else:
                image_binary[i, j] = 0
    return image_binary


image_gray = rgb_to_gray(image_rgb)
image_binary = gray_to_binary(image_gray)
cv.imshow('image_original', image)
cv.imshow('image_gray', image_gray)
cv.imshow('image_binary', image_binary)
cv.waitKey(0)