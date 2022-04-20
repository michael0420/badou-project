import numpy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img_path = "./4.png"
img = plt.imread(img_path)
# plt.imshow(img)
# plt.show()

imageR = img[:, :, 0]
imageG = img[:, :, 1]
imageB = img[:, :, 2]
print(imageR)
gray = imageR * 0.3 + imageG * 0.59 + imageB * 0.11
# plt.imshow(imageR, cmap='gray')
# plt.imshow(imageG, cmap='gray')
# plt.imshow(imageB, cmap='gray')
plt.imshow(gray, cmap='gray')
# plt.show()
# plt.show()
# plt.show()
plt.show()

img_binary = np.where(gray >= 0.5, 1, 0)
print(img_binary)
plt.imshow(img_binary,cmap='gray')
plt.show()


