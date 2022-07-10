import cv2
import numpy as np

img = cv2.imread('lenna.png')
scale = [800,800]

def Binlinear_interpolation(img,scale):
    img_h, img_w, img_c = img.shape
    dh, dw = scale[0] / img_h, scale[1] / img_w
    img_new = np.zeros([scale[0], scale[1], img_c], dtype=img.dtype)
    # 如果输出大小和变换后的大小相同则进行拷贝
    if img_h == scale[0] and img_w == scale[1]:
        return img.copy()
    for i in range(img_c):
        for j in range(scale[0]):
            for k in range(scale[1]):
                #进行变换使几何中心重合
                x = (j + 0.5) / dh - 0.5
                y = (k + 0.5) / dw - 0.5

                # 向下取整得到左端点，向右加1得到右端点
                x0 = int(np.floor(x))
                x1 = min(x0 + 1, img_h - 1)
                y0 = int(np.floor(y))
                y1 = min(y0 + 1, img_w - 1)

                #进行插值变换
                temp0 = (y1 - y) * img[x0, y0, i] + (y - y0) * img[x0, y1, i]
                temp1 = (y1 - y) * img[x1, y0, i] + (y - y0) * img[x1, y1, i]
                img_new[j, k, i] = int((x1 - x) * temp0 + (x - x0) * temp1)

    print('图像经过双线性插值变换后的值为{}'.format(img_new))
    cv2.imshow('Binlinear_interpolation_img', img_new)
    cv2.imwrite('Binlinear_interpolation_img.jpg', img_new)
    cv2.waitKey()
    return img_new

Binlinear_interpolation(img,scale)