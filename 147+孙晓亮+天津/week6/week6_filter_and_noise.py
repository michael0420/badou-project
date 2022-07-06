import numpy as np
import math
import matplotlib.pyplot as plt
import cv2 as cv
import random
import torch


def gauss_filter(img_path, sigma=0.5):
    # sigma = 1.52
    # sigma = 0.5
    dim = int(np.round(6 * sigma + 1))
    if dim % 2 == 0:
        dim += 1
    gaussian_filter = np.zeros([dim, dim])
    tmp = [i-dim//2 for i in range(dim)]
    n1 = 1/(2*math.pi*sigma**2)
    n2 = -1/(2*sigma**2)
    for i in range(dim):
        for j in range(dim):
            gaussian_filter[i, j] = n1*math.exp(n2*(tmp[i]**2+tmp[j]**2))
    gaussian_filter = gaussian_filter / gaussian_filter.sum()

    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    dx, dy, ch = img.shape
    img_new = np.zeros(img.shape)
    tmp = dim // 2
    img_pad = np.pad(img, ((tmp, tmp), (tmp, tmp), (0, 0)), 'constant')
    for c in range(ch):
        for i in range(dx):
            for j in range(dy):
                img_new[i, j, c] = np.sum(img_pad[i: i+dim, j: j+dim, c]*gaussian_filter)
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('original')
    plt.subplot(1, 2, 2)
    plt.imshow(img_new.astype(np.uint8))
    plt.axis('off')
    plt.title('gauss_filter_image(sigma={})'.format(sigma))
    plt.show()


def gauss_noise(img_path, means=2, sigma=4., percentage=0.8):
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    noise_img = img.copy()
    noise_num = int(img.shape[0] * img.shape[1] * percentage)
    for i in range(noise_num):
        rand_x = random.randint(0, img.shape[1]-1)
        rand_y = random.randint(0, img.shape[0]-1)
        # rand_c = random.randint(0, img.shape[2]-1)
        noise_img[rand_x, rand_y, :] = noise_img[rand_x, rand_y, :] + random.gauss(means, sigma)

    noise_img = torch.tensor(noise_img)
    noise_img = torch.clamp(noise_img, 0, 255)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('original')
    plt.subplot(1, 2, 2)
    plt.imshow(noise_img)
    plt.axis('off')
    plt.title('gauss_noise_image(means={}, sigma={})'.format(means, sigma))
    plt.show()


def pepper_salt_noise(img_path, percentage=0.8):
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    noise_img = img.copy()
    noise_num = int(img.shape[0] * img.shape[1] * percentage)
    for i in range(noise_num):
        rand_x = random.randint(0, img.shape[1] - 1)
        rand_y = random.randint(0, img.shape[0] - 1)
        rand_c = random.randint(0, img.shape[2]-1)

        if random.random() < 0.5:
            noise_img[rand_x, rand_y, rand_c] = 0
        else:
            noise_img[rand_x, rand_y, rand_c] = 255

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('original')
    plt.subplot(1, 2, 2)
    plt.imshow(noise_img)
    plt.axis('off')
    plt.title('pepper_salt_noise_image')
    plt.show()


if __name__ == '__main__':
    img_path = 'lena.png'
    # 高斯滤波
    # gauss_filter(img_path, sigma=2.23)
    # 高斯噪声
    # gauss_noise(img_path, means=2, sigma=4)
    # 椒盐噪声
    pepper_salt_noise(img_path, 1)
