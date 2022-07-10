# 最邻近插值， the nearest interpolation

import cv2
import numpy as np

def nearestInterp(img, target_height, target_width):
    height, width, channel = img.shape
    targetImage = np.zeros((target_height, target_width, channel), np.uint8)
    if height == target_height and width == target_width:
        return img.copy()
    h = target_height/height
    w = target_width/width
    for i in range(target_height):
        x = int(i/h)
        for j in range(target_width):
            y = int(j/w)
            targetImage[i][j] = img[x][y]
    return targetImage


img = cv2.imread("lenna.png")
target = nearestInterp(img,800,800)
print(target)
print(target.shape)
cv2.imshow("nearest interp",target)
cv2.imshow("image",img)
cv2.waitKey(0)