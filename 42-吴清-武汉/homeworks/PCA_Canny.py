# -*- coding:utf-8 -*-
# 八斗AI课作业.
# 特征提取降维算法-PCA&Canny
# 42-吴清-武汉
# 2022-05-15
import  numpy as np;
import tools.PCAHelper as helper
import matplotlib.pyplot as pl
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import tools.ImageHelper as tools
import cv2 as cv


def do():
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [11, 9,  35]])
    # PCA sklearn 方法
    # pca=PCA(n_components=2)
    # pca.fit(X);
    # new_X=pca.fit_transform(X)
    # print ("sklearn:",new_X)

    # PCA 手工实现
    #after_pca = helper.PCA(X,2)
    #print("after_pca:",after_pca)

    #Canny 实现
    imgs_canny = tools.image_canny("lenna.png")
    cv.imshow("canny-image", np.hstack(imgs_canny))
    cv.waitKey()
    cv.destroyAllWindows()