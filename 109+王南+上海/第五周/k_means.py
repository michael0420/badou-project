import numpy as np
import cv2
import matplotlib.pyplot as plt

src = cv2.imread("lena.jpeg")
data = np.float32(src.reshape((-1, 3)))
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
data_gray = np.float32(src_gray.reshape((-1, 1)))

k = 8
criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)
attempts = 3
flags = cv2.KMEANS_RANDOM_CENTERS

ret, labels, centers = cv2.kmeans(data, k, None, criteria, attempts, flags)
ret_gray, labels_gray, centers_gray = cv2.kmeans(data_gray, k, None, criteria, attempts, flags)

# labels = labels.reshape(src_gray.shape[0], src_gray.shape[1])
# for i in range(k):
#     src_gray[labels == i] = 255 * i // k
res = np.uint8(centers[labels.ravel()].reshape(src.shape))
res_gray = centers_gray[labels_gray.ravel()].reshape(src.shape[0], src.shape[1])


plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
plt.subplot(2, 2, 4)
# plt.imshow(src_gray, cmap="gray")
plt.imshow(res_gray, cmap="gray")
plt.show()
