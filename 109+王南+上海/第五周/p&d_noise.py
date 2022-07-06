import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


def pepper_salt_noise(src, percentage, ps_percentage):
    x, y = src.shape
    noise_num = int(x * y * percentage)
    for i in range(noise_num):
        rx = random.randint(0, x-1)
        ry = random.randint(0, y-1)
        if random.random() > ps_percentage:
            src[rx, ry] = 255
        else:
            src[rx, ry] = 0
    return src

src = cv2.imread("lena.jpeg")
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
src_ps = pepper_salt_noise(np.copy(src_gray), 0.2, 0.8)

plt.subplot(1, 2, 1)
plt.title("gray")
plt.axis("off")
plt.imshow(src_gray, cmap="gray")

plt.subplot(1, 2, 2)
plt.title("p&s")
plt.axis("off")
plt.imshow(src_ps, cmap="gray")

plt.show()