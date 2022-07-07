import cv2
import random
def GaussianNoise(src, means, sigma, percetage):
    NoiseNum = int( percetage * src.shape[0] * src.shape[1] )
    NoiseImg = src
    for i in range(NoiseNum):
        randx = random.randint(0,src.shape[0]-1)
        randy = random.randint(0,src.shape[1]-1)
        NoiseImg[randx, randy] = NoiseImg[randx, randy] + random.gauss(means, sigma)
        if NoiseImg[randx, randy]< 0:
            NoiseImg[randx, randy] = 0
        if NoiseImg[randx, randy] > 255:
            NoiseImg[randx, randy] =255
    return NoiseImg
img = cv2.imread('lenna.png', 0)
img1 = GaussianNoise(img, 2, 4, 0.8)
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('source',img2)
cv2.imshow('GaussianNoise', img1)
cv2.waitKey(0)