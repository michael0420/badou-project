import cv2
import random
def fun1(src, percetage):
    NoiseNum = int( percetage * src.shape[0] * src.shape[1] )
    NoiseImg = src
    for i in range(NoiseNum):
        randx = random.randint(0,src.shape[0]-1)
        randy = random.randint(0,src.shape[1]-1)
        if random.random()<=0.5:
            NoiseImg[randx, randy] = 0
        else:
            NoiseImg[randx, randy] = 255
    return NoiseImg
img = cv2.imread('lenna.png', 0)
img1 = fun1(img, 0.2)
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('source',img2)
cv2.imshow('PepperandSalt', img1)
cv2.waitKey(0)