import cv2 as cv2
import PIL.Image as im
import numpy as np
import  matplotlib.pyplot as plt
img=im.open(r'C:\Users\10128\Desktop\asd.jpg')

# print(type(img))
# print(img)
img_array=np.asarray(img)
# print(img_array)
print(img_array.shape)
g,k=img_array.shape[0:2]
print(g)
print(k)

# 转灰度

img_gray=np.zeros([g,k],img_array.dtype)
print(img_gray.shape)
for i in range(g):
    for j in range(k):
        m = img_array[i,j]
        img_gray[i,j]=int(m[0]*0.3+m[1]*0.59+m[2]*0.11)#bgr rgb
print("image show gray: %s"%img_gray)
print(img_gray.shape)

cv2.imshow("image show gray",img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()



# 转二值
img_b=np.zeros([g,k],dtype=img_array.dtype)

for i in range(g):
    for j in range(k):
        if(img_gray[i,j]<=112):
            img_b[i,j]=0
        else:
            img_b[i,j]=225
print(img_b)
print(img_b.shape)

cv2.imshow("二值",img_b)
cv2.waitKey(0)
cv2.destroyAllWindows()



# img=im.open(r'E:\百度网盘\八斗预习\八斗资料\回放资料\【2】数学基础&数字图像\代码\lenna.png')
# im.Image.show(img)
# img=cv.imread(r'C:\Users\10128\Desktop\asd.jpg',1)
# print(img.size)


