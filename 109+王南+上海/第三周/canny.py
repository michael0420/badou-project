import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

img = cv2.imread("lena.jpeg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dx, dy = img_gray.shape
plt.subplot(3, 2, 1)
plt.imshow(img_gray, cmap="gray")
plt.title("gray")
plt.axis("off")

sigma = 0.5
dim = int(round(sigma * 6 + 1))
if dim % 2 == 0:
    dim = dim + 1

gaussian = np.zeros((dim, dim), dtype=float)
tmp = [i - dim//2 for i in range(dim)]
for i in range(dim):
    for j in range(dim):
        gaussian[i, j] = (1 / (2 * math.pi * sigma**2)) * math.exp(-(tmp[i]**2 + tmp[j]**2) / (2 * sigma**2))
gaussian = gaussian / np.sum(gaussian)

img_gaussian = np.zeros(img_gray.shape)
img_gray_pad = np.pad(img_gray, ((dim//2, dim//2), (dim//2, dim//2)))

for i in range(dx):
    for j in range(dy):
        img_gaussian[i, j] = np.sum(img_gray_pad[i:i+dim, j:j+dim] * gaussian)

plt.subplot(3, 2, 2)
plt.imshow(img_gaussian, cmap="gray")
plt.title("gaussian")
plt.axis("off")

sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
img_sobel = np.zeros((dx, dy))
img_sobel_x = np.zeros((dx, dy))
img_sobel_y = np.zeros((dx, dy))
img_pad = np.pad(img_gaussian, ((1, 1), (1, 1)))

for i in range(dx):
    for j in range(dy):
        img_sobel_x[i, j] = np.sum(img_pad[i:i+3, j:j+3] * sobel_x)
        img_sobel_y[i, j] = np.sum(img_pad[i:i+3, j:j+3] * sobel_y)
        img_sobel[i, j] = np.sqrt(img_sobel_x[i, j]**2 + img_sobel_y[i, j]**2)

img_sobel_x[img_sobel_x == 0] = 0.00000001
angle = img_sobel_y / img_sobel_x

plt.subplot(3, 2, 3)
plt.imshow(img_sobel, cmap="gray")
plt.title("sobel")
plt.axis("off")

img_yizhi = np.zeros((dx, dy))
for i in range(1, dx-1):
    for j in range(1, dy-1):
        temp = img_sobel[i-1:i+2, j-1:j+2]
        if angle[i, j] <= -1:
            num1 = (temp[0, 1] - temp[0, 0]) / angle[i, j] + temp[0, 1]
            num2 = (temp[2, 1] - temp[2, 2]) / angle[i, j] + temp[2, 1]
            if img_sobel[i, j] > num1 and img_sobel[i, j] > num2:
                img_yizhi[i, j] = img_sobel[i, j]
        elif angle[i, j] >= 1:
            num1 = (temp[0, 2] - temp[0, 1]) / angle[i, j] + temp[0, 1]
            num2 = (temp[2, 0] - temp[2, 1]) / angle[i, j] + temp[2, 1]
            if img_sobel[i, j] > num1 and img_sobel[i, j] > num2:
                img_yizhi[i, j] = img_sobel[i, j]
        elif angle[i, j] > 0:
            num1 = (temp[0, 2] - temp[1, 2]) * angle[i, j] + temp[1, 2]
            num2 = (temp[2, 0] - temp[1, 0]) * angle[i, j] + temp[1, 0]
            if img_sobel[i, j] > num1 and img_sobel[i, j] > num2:
                img_yizhi[i, j] = img_sobel[i, j]
        elif angle[i, j] < 0:
            num1 = (temp[1, 0] - temp[0, 0]) * angle[i, j] + temp[1, 0]
            num2 = (temp[1, 2] - temp[2, 2]) * angle[i, j] + temp[1, 2]
            if img_sobel[i, j] > num1 and img_sobel[i, j] > num2:
                img_yizhi[i, j] = img_sobel[i, j]
plt.subplot(3, 2, 4)
plt.imshow(img_yizhi, cmap="gray")
plt.title("yizhi")
plt.axis("off")

lower = img_sobel.mean() * 0.5
high = lower * 3
stack = []
for i in range(1, img_yizhi.shape[0] - 1):
    for j in range(1, img_yizhi.shape[1] - 1):
        if img_yizhi[i, j] >= high:
            img_yizhi[i, j] = 255
            stack.append([i, j])
        elif img_yizhi[i, j] <= lower:
            img_yizhi[i, j] = 0

while not len(stack) == 0:
    temp_1, temp_2 = stack.pop()
    neighbour = img_yizhi[temp_1-1:temp_1+2, temp_2-1:temp_2+2]
    if (neighbour[0, 0] < high) and (neighbour[0, 0] > lower):
        img_yizhi[temp_1-1, temp_2-1] = 255  # 这个像素点标记为边缘
        stack.append([temp_1-1, temp_2-1])  # 进栈
    if (neighbour[0, 1] < high) and (neighbour[0, 1] > lower):
        img_yizhi[temp_1 - 1, temp_2] = 255
        stack.append([temp_1 - 1, temp_2])
    if (neighbour[0, 2] < high) and (neighbour[0, 2] > lower):
        img_yizhi[temp_1 - 1, temp_2 + 1] = 255
        stack.append([temp_1 - 1, temp_2 + 1])
    if (neighbour[1, 0] < high) and (neighbour[1, 0] > lower):
        img_yizhi[temp_1, temp_2 - 1] = 255
        stack.append([temp_1, temp_2 - 1])
    if (neighbour[1, 2] < high) and (neighbour[1, 2] > lower):
        img_yizhi[temp_1, temp_2 + 1] = 255
        stack.append([temp_1, temp_2 + 1])
    if (neighbour[2, 0] < high) and (neighbour[2, 0] > lower):
        img_yizhi[temp_1 + 1, temp_2 - 1] = 255
        stack.append([temp_1 + 1, temp_2 - 1])
    if (neighbour[2, 1] < high) and (neighbour[2, 1] > lower):
        img_yizhi[temp_1 + 1, temp_2] = 255
        stack.append([temp_1 + 1, temp_2])
    if (neighbour[2, 2] < high) and (neighbour[2, 2] > lower):
        img_yizhi[temp_1 + 1, temp_2 + 1] = 255
        stack.append([temp_1 + 1, temp_2 + 1])

for i in range(img_yizhi.shape[0]):
    for j in range(img_yizhi.shape[1]):
        if img_yizhi[i, j] != 0 and img_yizhi[i, j] != 255:
            img_yizhi[i, j] = 0

plt.subplot(3, 2, 5)
plt.imshow(img_yizhi, cmap="gray")
plt.title("result")
plt.axis("off")

plt.show()