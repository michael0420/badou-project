# encoding:utf-8

'''
第一次作业，调用opencv图像处理函数，实现图像灰度化和二值化
'''

import cv2
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文乱码
image_path = r'Lena.jpeg'
image1 = cv2.imread(image_path)[:, :, (2, 1, 0)] # 调换通道顺序

plt.subplot(131)
# imshow()对图像进行处理，画出图像，show()进行图像显示
plt.imshow(image1)

plt.title('原图')
# 不显示坐标轴
plt.axis('off')

# 子图2
image2 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
plt.subplot(132)
plt.imshow(image2)
plt.title('灰度化')
plt.axis('off')

# 子图3
# 图像的二值化
threshold = 80    # 二值化分割阈值
max_value = 255   # 图像中的最大值
[_, image3] = cv2.threshold(image1, threshold, max_value, cv2.THRESH_BINARY)

plt.subplot(133)
plt.imshow(image3)
plt.title('二值化')
plt.axis('off')

# #设置子图默认的间距
plt.tight_layout()
# 显示图像
plt.show()
