# -*- coding=utf-8 -*-

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot
from matplotlib import pyplot as plt


def get_sift_key_points(img):
    # 新建 sift 算子
    sift = cv2.xfeatures2d.SIFT_create()
    # 寻找特征点与特征点周围的特征向量
    key_points, descriptor = sift.detectAndCompute(img, None)

    img_plot = cv2.drawKeypoints(image=img, outImage=img, keypoints=key_points,
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=(51, 163, 236))
    # # cv2.imshow("sift keypoints", img_plot)
    plt.figure(figsize=(12, 8))
    plt.imshow(img_plot[:, :, ::-1])


def get_feature_match(img1, img2):
    sift = cv2.xfeatures2d.SIFT_create()
    box_key_points, box_descriptor = sift.detectAndCompute(img1, None)
    box_in_scene_key_point, box_in_scene_descriptor = sift.detectAndCompute(img2, None)
    # 创建 BF 对象
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matchers = bf.knnMatch(box_descriptor, box_in_scene_descriptor, k=2)

    good_matchers = list()
    for m, n in matchers:
        if m.distance < 0.5 * n.distance:
            good_matchers.extend([[m]])
    img_matches_knn = cv2.drawMatchesKnn(img1, box_key_points, img2, box_in_scene_key_point,
                                         good_matchers, None, flags=2)
    plt.figure(figsize=(12, 8))
    plt.imshow(img_matches_knn)