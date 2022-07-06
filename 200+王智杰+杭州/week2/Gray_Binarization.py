import cv2
import numpy as np

#读取需要处理的图像
img = cv2.imread('lenna.png')

#灰度化法1
def Gray1(img):
    img_h,img_w = img.shape[:2]
    img_gray = np.zeros([img_h,img_w] ,img.dtype)
    for i in range(img_h):
        for j in range(img_w):
            img_ = img[i,j]
            img_gray[i,j] = int(img_[0]*0.11 + img_[1]*0.59 + img_[2]*0.3) #将BGR三个通道值经过线性变换转化为gray坐标并赋值给新图像
    print('图像灰度化后的值为{}'.format(img_gray))
    cv2.imshow('Gray_img1',img_gray)
    cv2.imwrite('Gray_img1.jpg', img_gray)
    cv2.waitKey()
    return img_gray

#灰度化法2
def Gray2(img):
    #调用opencv库实现灰度化处理
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    print('图像灰度化后的值为{}'.format(img_gray))
    cv2.imshow('Gray_img2', img_gray)
    cv2.imwrite('Gray_img2.jpg', img_gray)
    cv2.waitKey()
    return img_gray

#二值化
def Binarization(Gray_img,threshold):
    Binarization_img = np.zeros_like(Gray_img,Gray_img.dtype)
    img_h, img_w = Gray_img.shape[:2]
    for i in range(img_h):
        for j in range(img_w):
            img_ = Gray_img[i,j]
            Binarization_img[i,j] = np.where(img_ >=threshold*255, 255,0)
    print('图像灰度化后的值为{}'.format(Binarization_img))
    cv2.imshow('Binarization_img',Binarization_img)
    cv2.imwrite('Binarization_img.jpg', Binarization_img)
    cv2.waitKey()
    return Binarization_img


# Gray1(img)
gray_img = Gray2(img)
#二值化阈值
threshold = 0.5
Binarization(gray_img,threshold)
