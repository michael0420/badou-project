#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
from matplotlib import pyplot as plt
import pandas as pd


# In[2]:


def nearest_interp(img, img_h, img_w, channels=3):
    img_nearest = np.zeros((img_h, img_w, channels),np.uint8)
    h_por = img_h/img.shape[0]
    w_por = img_w/img.shape[1]
    for i in range(img_h):
        for j in range(img_w):
            img_nearest[i,j] = img[int(i/h_por + 0.5), int(j/w_por + 0.5)]
    return img_nearest


# In[3]:


img = cv2.imread('preview.jpg')
img_h, img_w, channels = img.shape
print(img_h, img_w, channels)


# In[4]:


img_n = nearest_interp(img, 800, 800)


# In[5]:


cv2.imshow("nearest_interp",img_n)
cv2.imshow("image",img)
cv2.waitKey()
cv2.destroyAllWindows()


# In[ ]:




