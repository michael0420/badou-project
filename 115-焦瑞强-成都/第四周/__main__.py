# -*- coding=utf-8 -*-

import numpy as np
import cv2
from docs.conf import input_img_path
from bin.principal_component_analysis import get_principal_component
from bin.canny_edge_detection import get_canny_edge_detection

if __name__ == "__main__":
    ################################################################################################################
    #                                             PCA
    ################################################################################################################
    X = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9,  35],
                  [42, 45, 11],
                  [9,  48, 5],
                  [11, 21, 14],
                  [8,  5,  15],
                  [11, 12, 21],
                  [21, 20, 25]])
    K = np.shape(X)[1] - 1
    print(get_principal_component(X, K))

    ################################################################################################################
    #                                             Canny 边缘检测
    ################################################################################################################
    cv2.imshow("canny edge detection", get_canny_edge_detection(input_img_path, sigma=0.8).astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
