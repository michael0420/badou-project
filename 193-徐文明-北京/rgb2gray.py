import cv2
import numpy as np
from matplotlib import pyplot as plt

img = 'lenna.jpeg'

def rgbpic2gray(img, binvalue=False):
    picarr = cv2.imread(img)  # picarr 是bgr 不是rgb
    """
    灰度化公式 0.3r+0.59g+0.11b
    """
    gray = 0.3 * picarr[:, :, 2] + 0.59 * picarr[:, :, 1] + 0.11 * picarr[:, :, 0]
    gray = gray.astype(np.uint8)

    if binvalue:
        bingray = np.where(gray >= 127, 255, 0)
        bingray = bingray.astype(np.uint8)
        # print(bingray.min(),bingray.dtype)
        return bingray
    return gray

# 测试函数，显示图像
cv2.imshow('grey',rgbpic2gray(img,binvalue=False))
cv2.waitKey(0) # ms
cv2.destroyWindow(winname='grey')




