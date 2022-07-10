# -*- coding=utf-8 -*-

import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import seaborn as sns
import warnings

plt.style.use('ggplot')
warnings.filterwarnings('ignore')


def get_histogram_equalization(img):

    def get_hist_equal(pi_cum_sum, gray_level_num, gray_level_min, gray_level_max):
        hist_equal_pi = int(round(pi_cum_sum*gray_level_num - 1, 0))
        if hist_equal_pi < gray_level_min:
            return gray_level_min
        elif hist_equal_pi > gray_level_max:
            return gray_level_max
        else:
            return hist_equal_pi

    img_high, img_width = img.shape
    gray_level_unique, gray_level_counts = np.unique(img, return_counts=True)
    df_pix_ni = pd.DataFrame(np.asarray((gray_level_unique, gray_level_counts)).T, columns=["pix", "ni"])
    df_pix_ni["pi"] = df_pix_ni["ni"]/img.size
    df_pix_ni["pi_cum_sum"] = df_pix_ni["pi"].cumsum()

    df_pix_ni["hist_equal_pi"] = df_pix_ni.apply(
        lambda r: get_hist_equal(r["pi_cum_sum"], gray_level_unique.size, gray_level_unique[0], gray_level_unique[-1]),
        axis=1)

    df_original_img = pd.DataFrame(img.reshape((-1, 1)), columns=["pix"])
    df_original_img_hist = pd.merge(df_original_img, df_pix_ni, left_on=["pix"], right_on=["pix"], how="left")
    original_img_hist = df_original_img_hist["hist_equal_pi"].values.reshape((img_high, img_width)).astype(np.uint8)
    return original_img_hist


def get_plot_histogram(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = get_histogram_equalization(gray)

    ax1 = plt.subplot2grid((1, 2), (0, 0))
    sns.histplot(gray.ravel(), bins=256, color="steelblue", edgecolor="black")
    plt.title("Grayscale")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    ax2 = plt.subplot2grid((1, 2), (0, 1))
    sns.histplot(hist.ravel(), bins=256, color="steelblue", edgecolor="black")

    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.show()

    # fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(8, 3))
    # pic = sns.histplot(gray.ravel(), bins=256, color="steelblue", edgecolor="black")
    # pic.set_title("Grayscale")
    # pic = sns.histplot(hist.ravel(), bins=256, color="steelblue", edgecolor="black")
    # pic.set_title("Grayscale Histogram")
    # plt.show()

    cv2.imshow("Histogram Equalization", np.hstack([gray, hist]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()








