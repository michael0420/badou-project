import cv2 as cv2
import numpy as np


def img2gray(image):
    # print(img.shape)
    h, w, a = image.shape
    image_gray = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            image_gray[i][j] = (img[i][j][0] * 11 + img[i][j][1] * 59 + img[i][j][2] * 30) / 100
    return image_gray
    # print(image_gray.shape)
    # print(image_gray)


def gray2binary(image):
    h, w = image.shape
    image_binary = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if image[i][j] > 128:
                image_binary[i][j] = 255
            else:
                image_binary[i][j] = 0
    return image_binary


if __name__ == '__main__':
    path = 'lenna.png'
    img = cv2.imread(path)
    cv2.imshow("img", img)
    img_gray = img2gray(img)
    cv2.imshow("img_gray", img_gray)
    img_binary = gray2binary(img_gray)
    cv2.imshow("img_binary", img_binary)
    cv2.waitKey(0)
