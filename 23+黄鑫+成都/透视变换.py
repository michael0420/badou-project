import cv2
import numpy as np

img = cv2.imread("C:\\Users\\LENOVO\\Desktop\\lenna.png")

result3 = img.copy()

"""
输入原图像的点和dst图像的点
"""

src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst =np.flotat32([[0,0],[539,0],[0,623],[539,623]])

#生成透视变换矩阵，进行透视变换
m=cv2.getPerpectiveTransform(src,dst)
print("warpMatrix:")
print(m)
result = cv2.warpPerspective(result3,m,(337,488))
cv2.imshow("src",src)
cv2.imshow("result",result)
cv2.waitKey(0)
