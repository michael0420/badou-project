import imageio
import numpy as np
def RGB2Gray (file_path, save_name):
    # 读取图像
    img = imageio.imread(file_path)
    img = np.array(img)
    h, w, c = img.shape
    # 二值化
    gray_img = np.empty((h, w), dtype=int)
    for i in range(h):
        for j in range(w):
            # RGB平均值法
            gray_img[i, j] = int(np.mean(img[i, j, :]))
    # 保存图片
    imageio.imsave(save_name, gray_img)

def RGB2Binary(file_path, save_name):
    threshold = 135
    img = imageio.imread(file_path)
    img = np.array(img)
    h, w, c = img.shape
    gray_img = np.empty((h, w), dtype=int)
    for i in range(h):
        for j in range(w):
            if(np.mean(img[i, j, :]) > threshold):
                gray_img[i, j] = 255
            else:
                gray_img[i, j] = 0
    # 保存结果
    imageio.imsave(save_name, gray_img)