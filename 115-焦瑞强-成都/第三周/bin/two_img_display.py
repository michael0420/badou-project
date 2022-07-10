# -*- coding=utf-8 -*-

from cv2 import cv2
import numpy as np


def get_two_img_display(img1, img2, title):
    h0, w0 = img1.shape[0], img1.shape[1]
    h1, w1 = img2.shape[0], img2.shape[1]
    h = max(h0, h1)
    w = max(w0, w1)

    original_image = np.ones((h, w, 3), dtype=np.uint8) * 255
    transform_image = np.ones((h, w, 3), dtype=np.uint8) * 255

    original_image[:h0, :w0, :] = img1[:, :, :]
    transform_image[:h1, :w1, :] = img2[:, :, :]

    all_image = np.hstack((original_image[:, :w0, :], transform_image[:, :w1, :]))
    cv2.imshow(title, all_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



