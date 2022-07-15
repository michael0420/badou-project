# -*- coding=utf-8 -*-

import numpy as np
import cv2


def get_mean_hash(image):
    # 重置大小为 8*8
    image = cv2.resize(image, (8, 8), interpolation=cv2.INTER_CUBIC)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)
    # 计算均值
    image_mean = np.mean(image_gray)
    image_mean_hash = list(np.where(image_gray > image_mean, 1, 0).flat)
    return image_mean_hash


def get_diff_hash(image):
    # 重置大小为 8*9
    image = cv2.resize(image, (9, 8), interpolation=cv2.INTER_CUBIC)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(float)
    image_diff = np.diff(image_gray, axis=1)
    image_diff_hash = list(np.where(image_diff > 0, 1, 0).flat)
    return image_diff_hash


def get_hash_similarity(img_hash_1, img_hash_2):
    if len(img_hash_1) != len(img_hash_2):
        return -1
    else:
        img_hash_diff = np.asarray(img_hash_1) - np.asarray(img_hash_2)
        return img_hash_diff[np.where(img_hash_diff != 0)].size


if __name__ == "__main__":
    img_1 = cv2.imread(r"/R/week/week/data/input/img/aloeL.jpg")
    img_2 = cv2.imread(r"/R/week/week/data/input/img/aloeR.jpg")
    print(get_hash_similarity(get_mean_hash(img_1), get_mean_hash(img_2)))
