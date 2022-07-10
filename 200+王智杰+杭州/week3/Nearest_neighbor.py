import cv2
import numpy as np

#读取需要处理的图像
img = cv2.imread('lenna.png')
scale = [800,800]
def Nearest_neighbor(img,scale):
    img_h, img_w,img_c = img.shape
    dh, dw = scale[0]/ img_h , scale[1] / img_w
    img_new = np.zeros([scale[0],scale[1],img_c],dtype=np.uint8)
    #如果输出大小和变换后的大小相同则进行拷贝
    if img_h == scale[0] and img_w == scale[1]:
        return img.copy()
    for i in range(img_c):
        for j in range(scale[0]):
            for k in range(scale[1]):
                x = int((j / dh + 0.5))
                y = int((k / dw + 0.5))
                img_new[j,k,i] = img[x,y,i]
    print('图像经过最邻近插值变换后的值为{}'.format( img_new))
    cv2.imshow('Nearest_neighbor_img', img_new)
    cv2.imwrite('Nearest_neighbor_img.jpg', img_new)
    cv2.waitKey()
    return img_new

Nearest_neighbor(img,scale)

