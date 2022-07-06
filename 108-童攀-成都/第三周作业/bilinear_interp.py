#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt


# In[2]:


def bilinear_interp(img, img_h, img_w, channels=3):
    h_por = img_h/img.shape[0]
    w_por = img_w/img.shape[1]
    img_zero = np.zeros((img_h, img_w, channels), np.uint8)
    for channel in range(channels):
        for h in range(img_h):
            for w in range(img_w):
                scr_x = (w + 0.5)/h_por - 0.5
                scr_y = (h + 0.5)/w_por - 0.5

                scr_x0 = int(np.floor(scr_x + 0.5)) if int(np.floor(scr_x + 0.5))<img.shape[1]-1 else img.shape[1]-2
                scr_y0 = int(np.floor(scr_y + 0.5)) if int(np.floor(scr_y + 0.5))<img.shape[0]-1 else img.shape[0]-2
                scr_x1 = (scr_x0 + 1) if scr_x0 < scr_x else (scr_x0 -1)
                scr_y1 = (scr_y0 + 1) if  0< scr_y0 < scr_y else (scr_y0 -1)
                
                try:
                    if scr_x0 > scr_x:
                        temp0 = (scr_x0 - scr_x)*img[scr_y0, scr_x1, channel] + (scr_x - scr_x1)*img[scr_y0, scr_x0, channel]
                        temp1 = (scr_x0 - scr_x)*img[scr_y1, scr_x1, channel] + (scr_x - scr_x1)*img[scr_y1, scr_x0, channel]
                    else:
                        temp0 = (scr_x1 - scr_x)*img[scr_y0, scr_x0, channel] + (scr_x - scr_x0)*img[scr_y0, scr_x1, channel]
                        temp1 = (scr_x1 - scr_x)*img[scr_y1, scr_x0, channel] + (scr_x - scr_x0)*img[scr_y1, scr_x1, channel]

                    if scr_y0 > scr_y:
                        img_zero[h, w, channel] = (scr_y0 - scr_y)*temp1 +(scr_y - scr_y1)*temp0
                    else:
                        img_zero[h, w, channel] = (scr_y1 - scr_y)*temp0 +(scr_y - scr_y0)*temp1
                except Exception as e:
                    print(scr_x0, scr_y0, scr_x1, scr_y1)
    return img_zero


# In[3]:


img = cv2.imread("preview.jpg")
print(img.shape)


# In[4]:


img_b = bilinear_interp(img, 512, 512)


# In[5]:


cv2.imshow('orgin_img', img)
cv2.imshow('bilinear_interp', img_b)
cv2.waitKey()
cv2.destroyAllWindows()


# In[ ]:




