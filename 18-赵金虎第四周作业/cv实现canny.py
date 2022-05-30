import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('lenna.png')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_edg = cv.Canny(img, 100, 200)
plt.subplot(121), plt.imshow(img), plt.title('original image')
plt.subplot(122), plt.imshow(img_edg), plt.title('edge image')
plt.show()
