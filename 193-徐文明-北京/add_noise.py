import numpy as np
import random
import cv2
def add_gauss_noise(img,mu,sigma):
    m,n = img.shape
    for mi in range(m):
        for ni in range(n):
            img[mi,ni] = min(max(img[mi,ni]+random.gauss(mu,sigma),0),255)
    return img


def add_pepper_noise(img,snr):
    m,n = img.shape
    for mi in range(m):
        for ni in range(n):
            if random.random()<=snr:
                img[mi,ni] = random.choice([0,255])
    return img




img = cv2.imread('lenna.png',0)
print(img.shape)
# img = add_gauss_noise(img,30,10)
img = add_pepper_noise(img,0.3)
cv2.imshow('img',img)
cv2.waitKey()
cv2.destroyAllWindows()






