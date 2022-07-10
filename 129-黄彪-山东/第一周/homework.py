import numpy as np
import cv2

# 彩色图像灰度化
def color2Gray(img):
    h,w = img.shape[:2]
    img_gray = np.zeros([h,w], img.dtype)
    for i in range(h):
        for j in range(w):
            bgr = img[i,j]
            img_gray[i,j] = int(bgr[0] * 0.11 + bgr[1] * 0.59 + bgr[2] * 0.3)
    return img_gray

#彩色图像二值化
def color2Binary(img):
    img = color2Gray(img)
    h, w = img.shape[:2]
    img_bin = np.zeros([h, w], img.dtype)
    for i in range(h):
        for j in range(w):
            img_bin[i,j] = 0 if img[i,j] < 127 else 255
    return img_bin


if __name__ == '__main__':
    image = cv2.imread('lenna.png')
    # 显示原图像
    cv2.imshow('color image',image)
    #测试图像灰度化
    image1 = color2Gray(image)
    cv2.imshow('gray image',image1)
    #测试图像二值化
    image2 = color2Binary(image)
    cv2.imshow('binary image',image2)

    cv2.waitKey(0)
    # 关闭图片窗口
    cv2.destroyAllWindows()

