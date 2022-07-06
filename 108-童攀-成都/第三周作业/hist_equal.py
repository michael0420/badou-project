#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt


# In[2]:


img = cv2.imread("preview.jpg")


# In[3]:


channels = cv2.split(img)
colors = ('b', 'g', 'r')





# In[5]:


color_B = cv2.equalizeHist(channels[0])
color_G = cv2.equalizeHist(channels[1])
color_R = cv2.equalizeHist(channels[2])
img_equal = cv2.merge((color_B,color_G, color_R))
cv2.imshow('orgin_img',img)
cv2.imshow('img_equal',img_equal)
cv2.waitKey()
cv2.destroyAllWindows()


# In[6]:


channels_equal = cv2.split(img_equal)
plt.figure(figsize=(12,5))
plt.subplot2grid((1, 2), (0, 0), colspan=1)
for (channel, color) in zip(channels, colors):
    hist = cv2.calcHist([channel],[0],None,[256], [0,256])
    plt.plot(hist, label=color)
plt.title('color_hist')
plt.xlabel('grey level')
plt.ylabel('number')
plt.subplot2grid((1, 2), (0, 1), colspan=1)
for (channel, color) in zip(channels_equal, colors):
    hist = cv2.calcHist([channel],[0],None,[256], [0,256])
    plt.plot(hist, label=color)
plt.title('color_equal')
plt.xlabel('grey level')
plt.ylabel('number')
plt.legend()
plt.show()



