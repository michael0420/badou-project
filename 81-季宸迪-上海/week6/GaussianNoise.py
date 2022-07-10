import cv2
import random

def gaussianNoise(src, means, sigma, percentage):
    noiseImg = src
    num = int(percentage*src.shape[0]*src.shape[1])
    visited = set()
    count = 0
    while count < num:
        randX = random.randint(0, src.shape[0]-1)
        randY = random.randint(0, src.shape[1]-1)
        if (randX, randY) in visited:
            continue
        noiseImg[randX][randY] += random.gauss(means, sigma)
        if noiseImg[randX][randY] < 0:
            noiseImg[randX][randY] = 0
        if noiseImg[randX][randY] > 255:
            noiseImg[randX][randY] = 255
        count += 1
        visited.add((randX,randY))
    return noiseImg

img = cv2.imread('lenna.png',0)
img1 = gaussianNoise(img,2,4,0.8)
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imwrite('lenna_GaussianNoise.png',img1)
cv2.imshow('source',img2)
cv2.imshow('lenna_GaussianNoise',img1)
cv2.waitKey(0)