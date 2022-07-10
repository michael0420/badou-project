import random
import cv2

def pepperSalt(src, percentage):
    noiseImg = src
    num = int(percentage*src.shape[0]*src.shape[1])
    visited = set()
    count = 0
    while count < num:
        randX = random.randint(0, src.shape[0]-1)
        randY = random.randint(0, src.shape[1]-1)
        if (randX, randY) in visited:
            continue
        if random.random() <= 0.5:
            noiseImg[randX, randY] = 0
        else:
            noiseImg[randX, randY] = 255
        visited.add((randX, randY))
        count += 1
    return noiseImg

img = cv2.imread('lenna.png',0)
img1 = pepperSalt(img,0.2)
#在文件夹中写入命名为lenna_PepperSalt.png的加噪后的图片
#cv2.imwrite('lenna_PepperSalt.png',img1)

img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('source',img2)
cv2.imshow('lenna_PepperSalt',img1)
cv2.waitKey(0)
