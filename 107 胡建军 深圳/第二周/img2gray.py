import cv2
import numpy as np
import matplotlib.pyplot as plt

def img2gray(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    gray = 0.3*R + 0.59*G + 0.11*B
    return gray

def img2binary(img):
    b_img = np.where(img>127, 255, 0)
    return b_img

def read_img(path):
    img = cv2.imread(path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#bgr2rgb
        print(img.shape)
    else:
        print('图片数据为空')
    return img

if __name__ == '__main__':
    path = './lenna.png'
    img = read_img(path)#已转为RGB格式

    #cv2库函数
    cv2gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imshow('gray', cv2gray_img)
    # bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    retval, cv2binary_img = cv2.threshold(cv2gray_img, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow('binary', cv2binary_img)

    # print(gray_img)
    #自定义
    gray_img = img2gray(img).astype(int)
    binary_img = img2binary(gray_img)
    plt.figure(num='lenna', figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
    #显示原始图像
    plt.subplot(221)
    plt.title('Original Image')
    plt.xticks([]);plt.yticks([])
    plt.imshow(img)
    #显示灰度图像
    plt.subplot(222)
    plt.title('Gray Image')
    plt.xticks([]);plt.yticks([])
    plt.imshow(gray_img, cmap='gray')
    #显示二值图像
    plt.subplot(223)
    plt.title('Binary Image')
    plt.xticks([]);plt.yticks([])
    plt.imshow(binary_img, cmap='gray')
    plt.show()

    cv2.waitKey()


