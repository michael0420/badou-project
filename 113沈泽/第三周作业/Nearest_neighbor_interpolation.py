import cv2
import numpy as np

def function(img, h, w):
    height, width, channels = img.shape
    emptyImage = np.zeros((h, w, channels), np.uint8)
    sh = h/height
    sw = w/width
    for i in range(h):
        for j in range(w):
            x = int((i/sh)+0.5)
            y = int((j/sw)+0.5)
            emptyImage[i, j] = img[x, y]
    return emptyImage

img = cv2.imread("lenna.png")
zoom = function(img, 800, 800)
cv2.imshow("Nearest_neighbor_interpolation", zoom)
cv2.imshow("img", img)
cv2.waitKey(0)
