"""
work_2_1:  实现最近邻插值
运用包/模块：numpy、cv2
"""
import cv2
import numpy as np
def function_nearest_interp(img,out_dim):
    height,width,channels = img.shape
    emptyImage = np.zeros(out_dim,np.uint8)
    sh = height/out_dim[0]
    sw = width/out_dim[1]
    for i in range(out_dim[0]):
        for j in range(out_dim[1]):
            x = i * sh
            y = j * sw
            # 将x和y进行向下取整，得到原图上对应的像素位置(scrX, srcY)
            ix = int(x)
            iy = int(y)
            # 计算目标像素与原图像上整数像素之间的距离
            u = x - ix
            v = y - iy
            # 根据距离来判断该选择周围四个像素中哪个像素
            if u > 0.5:
                ix += 1
            if v > 0.5:
                iy += 1
            emptyImage[i,j] = img[ix,iy]
    return  emptyImage
img = cv2.imread("sunset.jpg")
zoom = function_nearest_interp(img,[800,1200,3])
print(zoom)
print(zoom.shape)
print(img.shape)
cv2.imshow("nearest interp",zoom)
cv2.imshow("original image",img)
cv2.waitKey(0)
