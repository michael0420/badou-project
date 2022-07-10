import numpy as np
import matplotlib.pyplot as plt
import cv2

# 原图(BGR)
image_BGR = cv2.imread("lenna.png")
h = image_BGR.shape[0]
w = image_BGR.shape[1]
plt.subplot(2, 2, 1)
plt.imshow(image_BGR)
plt.axis("off")     # 关闭坐标轴
plt.title("lenna_BGR")

# 原图(RGB)
image = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
plt.subplot(2, 2, 2)
plt.imshow(image)
plt.axis("off")     # 关闭坐标轴
plt.title("lenna_RGB")

# 灰度化
image_gray = np.zeros((h, w), dtype=image.dtype)
for i in range(h):
    for j in range(w):
        current_pixel = image[i][j]
        image_gray[i][j] = int(current_pixel[0] * 0.3 + current_pixel[1] * 0.59 + current_pixel[2] * 0.11)
        # print(image_gray[i][j])
# cv2.imshow("lenna_gray", image_gray)
# cv2.waitKey(0)
plt.subplot(2, 2, 3)
# 这里必须加 cmap='gray', 否则尽管原图像是灰度图，但是显示的是伪彩色图像
plt.imshow(image_gray, cmap="gray")
plt.axis("off")     # 关闭坐标轴
plt.title("lenna_gray")
cv2.imwrite("./lenna_gray.png", image_gray)

# 二值化
image_binary = np.zeros((h, w), dtype=image.dtype)
for i in range(h):
    for j in range(w):
        if image_gray[i][j] > 142:
            image_binary[i][j] = 255
        else:
            image_binary[i][j] = 0
# cv2.imshow("lenna_binary", image_binary)
# cv2.waitKey(0)
plt.subplot(2, 2, 4)
plt.imshow(image_binary, cmap="gray")
plt.axis("off")     # 关闭坐标轴
plt.title("lenna_binary")
plt.show()
cv2.imwrite("./lenna_binary.png", image_binary)
