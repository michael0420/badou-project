from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 获取图片
def getimg():
  return Image.open("lenna.png")
  
# 显示图片
def showimg(img, isgray=False):
  plt.axis("off")
  if isgray == True:
    plt.imshow(img, cmap='gray')
  else: 
    plt.imshow(img)
  plt.show()

# PIL库自带函数实现
im = getimg()
im_gray = im.convert('L')
showimg(im_gray, True)

# 浮点数算法实现
im = getimg()
im = np.array(im)
im[:,:,0] = im[:,:,0]*0.3    # R
im[:,:,1] = im[:,:,1]*0.59   # G
im[:,:,2] = im[:,:,2]*0.11   # B
im = np.sum(im, axis=2)
showimg(Image.fromarray(im), True)

# 整数算法实现
im1 = getimg()
#创建数组时指定数据类型，否则默认uint8乘法运算会溢出
im1 = np.array(im1, dtype=np.float32)
im1[...,0] = im1[...,0]*40.0
im1[...,1] = im1[...,1]*70.0
im1[...,2] = im1[...,2]*15.0
im1 = np.sum(im1, axis=2)
im1[...,:] = im1[...,:]/100.0
showimg(Image.fromarray(im1), True)

# 二值化,取127作为阀值
im2 = getimg()
im2 = np.array(im2.convert('L'))
im2 = np.where(im2[...,:] < 127, 0, 255)
showimg(Image.fromarray(im2), True)