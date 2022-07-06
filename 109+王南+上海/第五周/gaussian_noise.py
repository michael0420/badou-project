import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

def add_gaussian_noise(src, mu, sigma, percentage):
    x, y = src.shape
    noise_num = int(x * y * percentage)
    gaussian_random_num = random.gauss(mu, sigma)
    for i in range(noise_num):
        rx = random.randint(0, x - 1)
        ry = random.randint(0, y - 1)
        src[rx, ry] = src[rx, ry] + gaussian_random_num
    return src

src = cv2.imread("lena.jpeg")
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
src_gaussian = add_gaussian_noise(np.copy(src_gray), 20, 50, 0.8)

plt.subplot(1, 2, 1)
plt.title("gray")
plt.axis("off")
plt.imshow(src_gray, cmap="gray")

plt.subplot(1, 2, 2)
plt.title("gaussian")
plt.axis("off")
plt.imshow(src_gaussian, cmap="gray")

plt.show()