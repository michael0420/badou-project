#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from matplotlib import pyplot as plt


# In[2]:


img = cv2.imread('lenna.png')
print(img.shape)


# In[3]:


img_h, img_w = img.shape[0], img.shape[1]
print('img_h:', img_h)
print('img_w:', img_w)


# In[4]:


img_gray1 = np.zeros((img_h, img_w))
print(img_gray1.shape)



# In[5]:


print(img[0, 0])



# In[6]:


img_original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_original)


# In[7]:


# 灰度化
for i in range(img_h):
    for j in range(img_w):
        img_gray1[i, j] = img[i, j][0]*0.11 + img[i, j][1]*0.59 + img[i, j][2]*0.3


# In[8]:


print(img_gray1.shape)



# In[9]:


plt.imshow(img_gray1, cmap='gray')


# In[10]:


# opencv接口灰度化
img_gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img_gray2.shape)



# In[11]:


plt.imshow(img_gray2, cmap='gray')


# In[ ]:





# In[12]:


img_binary1 = np.zeros((img_h, img_w))
print(img_binary1.shape)



# In[13]:


# 二值化

for i in range(img_h):
    for j in range(img_w):
        img_binary1[i, j] = 1 if img_gray1[i, j]/255 >= 0.5 else 0


# In[14]:


print(img_binary1)



# In[15]:


plt.imshow(img_binary1, cmap='gray')


# In[16]:


# numpy二值化
img_binary2 = np.where(img_gray2/255>0.5, 1, 0)
print(img_binary2)


# In[17]:


plt.imshow(img_binary2, cmap='gray')


# In[18]:


fig = plt.figure(figsize=(10,10))
plt.subplot2grid((2,2), (0,0), colspan=1)
plt.imshow(img_original)
plt.title('lenna')
plt.subplot2grid((2,2), (0,1), colspan=1)
plt.imshow(img_gray1, cmap='gray')
plt.title('lenna_gray')
plt.subplot2grid((2,2), (1,0), colspan=1)
plt.imshow(img_binary1, cmap='gray')
plt.title('lenna_binary')
plt.show()


# In[ ]:





# In[ ]:




