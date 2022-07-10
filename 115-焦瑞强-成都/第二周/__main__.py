# -*- coding=utf-8 -*-

from docs.conf import input_img_path
from bin.rgb_gray_binary import get_img_rgb_gray, get_img_binary

if __name__ == "__main__":
    get_img_rgb_gray(input_img_path)
    get_img_binary(input_img_path)
