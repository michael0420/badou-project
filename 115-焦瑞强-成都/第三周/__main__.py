# -*- coding=utf-8 -*-

import cv2
from docs.conf import input_img_path
from bin.nearest_neighbor_bilinear_interpolation import get_nearest_neighbor_interpolation, get_bilinear_interpolation
from bin.histogram_equalization import get_plot_histogram
from bin.two_img_display import get_two_img_display

if __name__ == "__main__":
    img = cv2.imread(input_img_path)
    nearest_neighbor_img = get_nearest_neighbor_interpolation(img, 800, 800)
    get_two_img_display(img, nearest_neighbor_img, "nearest interpolation")

    bilinear_img = get_bilinear_interpolation(img, 800, 800)
    get_two_img_display(img, bilinear_img, "bilinear interpolation")

    get_plot_histogram(img)
