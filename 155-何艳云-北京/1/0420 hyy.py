#coding:utf8
import cv2

#灰度化
def Image_gray(image):
    h, w, ch = image.shape
    for row in range(h):
        for col in range(w):
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            img_gray_f = int(0.11 * b + 0.59 * g + 0.3 * r)
            image[row, col, 0] = img_gray_f
            image[row, col, 1] = img_gray_f
            image[row, col, 2] = img_gray_f

    cv2.imshow("image show in gray", image)
    cv2.waitKey(1000)
    cv2.imwrite("gray.png", image)


img = cv2.imread('D:\AI hyy\lenna.png')
Image_gray(img)

#二值化
img2 = cv2.imread('D:\AI hyy\lenna.png', 0)
cv2.imshow("image show in gray", img2)
cv2.waitKey(1000)
img2 = img2/255      #归一化
cv2.imshow("Normalization", img2)
cv2.waitKey(1000)
h, w = img2.shape
for row in range(h):
    for col in range(w):
      threshold = 0.5     #阈值设定
      if (img2[row][col] <= threshold):
          img2[row][col]  = 0
      else: img2[row][col] = 1
cv2.imshow("image show in blackwhite", img2)
cv2.waitKey(1000)
cv2.imwrite("blackwhite.png", img2)


#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

