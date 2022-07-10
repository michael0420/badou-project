# -*- coding=utf-8 -*-

import numpy as np
import cv2
from docs.conf import input_img_path
from bin.perspective_transformation import get_img_preprocess, get_vertex, get_order_points,get_target_point, get_warp_matrix


if __name__ == "__main__":
    img_original = cv2.imread(input_img_path)
    img_high, img_width = img_original.shape[0], img_original.shape[1]
    img_ = get_img_preprocess(img_original)
    img_hough_lines = cv2.HoughLinesP(img_, rho=1.1, theta=np.pi / 180, threshold=300, minLineLength=200, maxLineGap=1)
    vertex_list = get_vertex(img_hough_lines)
    vertex_list = list(set(filter(lambda i: i[0] < img_width and i[1] < img_high, vertex_list)))
    original_vertex = get_order_points(vertex_list)
    target_vertex = get_target_point(original_vertex)
    perspective_matrix = get_warp_matrix(original_vertex, target_vertex)
    perspective_img = cv2.warpPerspective(img_original, perspective_matrix,
                                          (int(original_vertex[2][0]), int(target_vertex[2][1])))

    original_perspective_img = np.hstack((img_original, perspective_img))
    cv2.imshow("perspective img", original_perspective_img)
    cv2.waitKey(0)
