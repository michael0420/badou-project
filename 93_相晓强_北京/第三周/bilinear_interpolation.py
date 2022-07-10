#双线性插值算法（上采样）   完成
import numpy as np
import cv2
 
'''
python implementation of bilinear interpolation  Python实现的双线性插值
''' 
def bilinear_interpolation(img,out_dim):
    src_h, src_w, channel = img.shape           #获得原图像的高、宽、通道
    dst_h, dst_w = out_dim[1], out_dim[0]       #获得目标图像的高、宽     前宽后高！
    print ("src_h, src_w = ", src_h, src_w)
    print ("dst_h, dst_w = ", dst_h, dst_w)     #输出尺寸
    if src_h == dst_h and src_w == dst_w:       
        return img.copy()                       #如果相同则复制原图像给目标图像 
    dst_img = np.zeros((dst_h,dst_w,3),dtype=np.uint8)                #创建一个等同目标图像大小的全零数组
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h     #获得比例
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
 
                # find the origin x and y coordinates of dst image x and y   求DST图像x和y的原点x和y坐标
                # use geometric center symmetry
                # if use direct way, src_x = dst_x * scale_x   如果使用直接的方式
                src_x = (dst_x + 0.5) * scale_x-0.5      #几何中心对称
                src_y = (dst_y + 0.5) * scale_y-0.5
 
                # find the coordinates of the points which will be used to compute the interpolation 找出将被用来计算插值的点的坐标
                src_x0 = int(np.floor(src_x))    #floor 向下取整 ？？？？？
                src_x1 = min(src_x0 + 1 ,src_w - 1)     
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)
 
                # calculate the interpolation   计算插值？？？？？？？？？
                temp0 = (src_x1 - src_x) * img[src_y0,src_x0,i] + (src_x - src_x0) * img[src_y0,src_x1,i]
                temp1 = (src_x1 - src_x) * img[src_y1,src_x0,i] + (src_x - src_x0) * img[src_y1,src_x1,i]
                dst_img[dst_y,dst_x,i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
 
    return dst_img
 
 
if __name__ == '__main__':   #import 到其他脚本中是不会被执行的
    img = cv2.imread('lenna.png')                #图像读取
    dst = bilinear_interpolation(img,(600,900))  #调用函数
    cv2.imshow('bilinear interp',dst)            #图像显示
    cv2.imshow("123", img)
    cv2.waitKey()
