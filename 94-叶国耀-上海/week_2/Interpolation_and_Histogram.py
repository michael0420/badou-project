import cv2
import numpy as np
from matplotlib import pyplot as plt

def nearest_interp(img, out_h, out_w):
    height, width, channels = img.shape
    if height == out_h and width == out_w:
        return img.copy()
    img_out = np.zeros((out_h, out_w, channels), np.uint8)
    sh = height / out_h
    sw = width / out_w
    for i in range(out_h):
        for j in range(out_w):
            x = int(i * sh) + 1 if (i * sh >= (int(i * sh) + 0.5)) else int(i * sh)
            y = int(j * sw) + 1 if (j * sw >= (int(j * sw) + 0.5)) else int(j * sw)
            img_out[i, j] = img[x, y]

    return img_out

def bilinea_interp(img, out_h, out_w):
    height, width, channels = img.shape
    if height == out_h and width == out_w:
        return img.copy()
    img_out = np.zeros((out_h, out_w, channels), np.uint8)
    sh = height / out_h
    sw = width / out_w
    for i in range(3):
        for h in range(out_h):
            for w in range(out_w):
                src_x = (h + 0.5) * sh - 0.5
                src_y = (w + 0.5) * sw - 0.5

                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, height - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, width - 1)

                temp0 = (src_x1 - src_x) * img[src_x0, src_y0, i] + (src_x - src_x0) * img[src_x1, src_y0, i]
                temp1 = (src_x1 - src_x) * img[src_x0, src_y1, i] + (src_x - src_x0) * img[src_x1, src_y1, i]
                img_out[h, w, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    return img_out

img = cv2.imread("lenna.png")
cv2.imshow("origin_image",img)

img_near = nearest_interp(img, 800, 800)
cv2.imshow("nearest_interp_image",img_near)

img_near = bilinea_interp(img, 800, 800)
cv2.imshow("bilinea_interp_image",img_near)

cv2.waitKey(0)

# 直方图
chans = cv2.split(img)
colors = ("b","g","r")
plt.figure()
plt.title("Flattened Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

for (chan,color) in zip(chans,colors):
    hist = cv2.calcHist([chan],[0],None,[256],[0,256])
    plt.plot(hist,color = color)
    plt.xlim([0,256])
plt.show()
# 直方图均衡
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((bH, gH, rH))
chans = cv2.split(result)
cv2.imshow("dst_rgb", result)
for (chan,color) in zip(chans,colors):
    hist = cv2.calcHist([chan],[0],None,[256],[0,256])
    plt.plot(hist,color = color)
    plt.xlim([0,256])
plt.show()
cv2.waitKey(0)