# -*- coding=utf-8 -*-


import numpy as np
import cv2


def get_img_rgb_gray(img_path, red_weight=0.3, blue_weight=0.11, green_weight=0.59, is_display=1):
    """
    :param img_path:
    :param red_weight:
    :param blue_weight:
    :param green_weight:
    :param is_display:
    :return:
    """
    img = cv2.imread(img_path, 1)
    cv2.imshow("img", img)
    rows, columns, channels = img.shape
    gray_img = np.zeros([rows, columns], dtype=img.dtype)
    bgr_weight = np.array([blue_weight, green_weight, red_weight]).reshape((-1, 1))
    for r in range(rows):
        gray_img[r, ] = np.apply_along_axis(lambda p: np.dot(p, bgr_weight), axis=1, arr=img[r]).reshape((1, -1))
    if is_display:
        cv2.imshow("gray_img", gray_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        return gray_img


def get_img_binary(img_path, threshold=127):
    """
    :param img_path:
    :param threshold:
    :return:
    """
    gray_img = get_img_rgb_gray(img_path=img_path, is_display=0)
    binary_img = np.where(gray_img > threshold, np.uint8(255), np.uint8(0))
    cv2.imshow("binary_img", binary_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
