# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 17:36:59 2022

@author: Administrator
"""

import matplotlib.pyplot as plt
import numpy as np

#%% 实现灰度化
img = plt.imread('lenna.png')
h,w = img.shape[:2]
img_gray = np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i,j]
        img_gray[i,j] = m[0] * 0.3 + m[1] * 0.59 + m[2] * 0.11
        
        
#%% 实现二值化
img_binary = np.where(img_gray >= 0.5, 1, 0)

#%% 图片展示

plt.subplot(131)
plt.imshow(img)
plt.subplot(132)
plt.imshow(img_gray, cmap='gray')
plt.subplot(133)
plt.imshow(img_binary, cmap='gray')
