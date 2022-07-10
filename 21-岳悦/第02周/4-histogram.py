import matplotlib.pyplot as plt
import cv2
import os
imagepath = os.getcwd()
img = cv2.imread("./image/lenna.png", 1)
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(rgb_img)
plt.show()

(r, g, b) = cv2.split(rgb_img)
rH = cv2.equalizeHist(r)
plt.hist(rH.ravel(),256)
plt.show()

gH = cv2.equalizeHist(g)
plt.hist(gH.ravel(),256)
plt.show()

bH = cv2.equalizeHist(b)
plt.hist(bH.ravel(),256)
plt.show()

result = cv2.merge((rH, gH, bH))

plt.imshow(result)
plt.show()
