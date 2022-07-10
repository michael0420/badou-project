# -*- coding=utf-8 -*-


import numpy as np
import random


def get_salt_pepper_noise(img, ratio):
    high = img.shape[0]
    width = img.shape[1]
    res = np.zeros((high, width), dtype=np.uint8)
    for i in range(high):
        for j in range(width):
            rand = random.random()
            if rand < ratio:
                res[i, j] = 0 if random.random() < 0.5 else 255
            else:
                res[i, j] = img[i, j]

    return res


def get_gaussian_noise(img, ratio, sigma):

    def get_gaussian_kernel(x, mu, sigma):
        return np.exp(-1 * np.power(x - mu, 2) / (2 * np.power(sigma, 2))) / (np.sqrt(2 * np.pi) * sigma)

    high = img.shape[0]
    width = img.shape[1]
    res = np.zeros((high, width), dtype=np.uint8)
    x = np.linspace(-4 * sigma, 4 * sigma, 100)
    gaussian_kernel = get_gaussian_kernel(x, 0, sigma)

    for h in range(high):
        for w in range(width):
            rand = random.random()
            if rand < ratio:
                index = int(random.random() * 100)
                if x[index] < 0:
                    res[h, w] = img[h, w] * (1 - gaussian_kernel[index])
                else:
                    res[h, w] = img[h, w] * (1 - gaussian_kernel[index])
            else:
                res[h, w] = img[h, w]
    return np.clip(res, 0, 255)
