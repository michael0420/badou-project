import cv2
import numpy as np

def function(img):
    height,width,channels =img.shape  #返回图片的长、宽、通道
    scale_h = float(height/800)
    scale_w = float(width/800)
    print(height)
    print(scale_w)
    dst_img = np.zeros((800, 800, 3), dtype=np.uint8)

    for channel in range(channels):
        for dst_h in range(800):
            for dst_w in range(800):
                # src_h = dst_h * scale_h
                # src_w = dst_w * scale_w  普通放大

                src_h = (dst_h + 0.5) * scale_h - 0.5
                src_w = (dst_w + 0.5) * scale_w - 0.5   #中心放大

                src_x0 = int(src_w)
                src_x1 = min(src_x0 + 1, width - 1)
                src_y0 = int(src_h)
                src_y1 = min(src_y0 + 1, height - 1)

                dst_y0 = (src_x1 - src_w)*img[src_y0,src_x0,channel] + (src_w - src_x0) * img[src_y0, src_x1, channel]
                dst_y1 = (src_x1 - src_w) * img[src_y1, src_x0, channel] + (src_w - src_x0) * img[
                    src_y1, src_x1, channel]

                dst_img[dst_h, dst_w, channel] = (src_y1 - src_h)*dst_y0 + (src_h - src_y0) * dst_y1


    return dst_img



img = cv2.imread("img/lenna.png")
img_BiliInter = function(img)

cv2.imshow("bilinear Interpolation",img_BiliInter)
cv2.imshow("image",img)
cv2.waitKey(0)
