import cv2
import matplotlib.pyplot as plt
import numpy as np

src = cv2.imread("gezi.jpeg")
x, y, z = src.shape
print(x, y, z)

matrix = cv2.getPerspectiveTransform(
    # np.float32([[0, 0], [499, 0], [0, 331], [499, 331]]),
    np.float32([[89, 53], [90, 282], [122, 52], [122, 282]]),

    # np.float32([[89, 52], [89, 280], [419, 52], [419, 280]])
    # np.float32([[122, 2], [3, 116], [355, 3], [387, 330]])
    np.float32([[58, 55], [57, 282], [105, 54], [105, 282]])
)

dst = cv2.warpPerspective(src, matrix, (y, x))
# dst1 = cv2.perspectiveTransform(src, matrix, (y, x))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))

# plt.subplot(2, 2, 4)
# plt.imshow(cv2.cvtColor(dst1, cv2.COLOR_BGR2RGB))

plt.show()

