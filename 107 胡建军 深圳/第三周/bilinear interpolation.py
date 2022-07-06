import cv2
import numpy as np

def read_img(path):
    img = cv2.imread(path)
    return img

def bilinear_interpolation(img, out_dim):
    sh, sw, c = img.shape
    dh, dw = out_dim[0], out_dim[1]
    dst_img = np.zeros([dh, dw, c], dtype=np.uint8)
    kx = sw / dw
    ky = sh/dh
    for i in range(c):
        for dy in range(dh):
            for dx in range(dw):
                sx = (dx + 0.5)*kx - 0.5#>-0.5
                sy = (dy + 0.5)*ky - 0.5
                sx0 = int(sx)#最小为0
                sx1 = min(sx0+1, sw-1)
                sy0 = int(sy)
                sy1 = min(sy0+1, sh-1)
                temp0 = (sx1-sx)*img[sy0, sx0, i] + (sx-sx0)*img[sy0, sx1, i]
                temp1 = (sx1-sx)*img[sy1, sx0, i] + (sx-sx0)*img[sy1, sx1, i]
                dst_img[dy, dx, i] = round((sy1-sy)*temp0 + (sy-sy0)*temp1)

    return dst_img



if __name__ == '__main__':
    path = './lenna.png'
    img = read_img(path)
    resize_img = bilinear_interpolation(img, (1200, 1200))
    cv2.imshow('original img', img)
    cv2.imshow('bilinear interp',resize_img)
    cv2.waitKey()