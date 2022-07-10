# -*-coding=utf-8 -*-

import numpy as np


def get_nearest_neighbor_interpolation(img, destination_high, destination_width):
    source_high, source_width, source_channel = img.shape
    destination_img = np.zeros((destination_high, destination_width, source_channel), dtype=np.uint8)
    for x in range(destination_high):
        for y in range(destination_width):
            source_x = int(round(x * source_high / destination_high))
            source_y = int(round(y * source_width / destination_width))
            destination_img[x, y] = img[source_x, source_y]
    return destination_img


def get_bilinear_interpolation(img, destination_high, destination_width):

    def interpolate_values(x_0, x_1, x, f_0, f_1):
        return np.dot(np.array([x_1 - x, x - x_0]), np.array([f_0, f_1]).T)

    source_high, source_width, source_channel = img.shape
    if destination_high == source_high and destination_width == source_width:
        return img.copy()
    else:
        scale_x = source_width / float(destination_width)
        scale_y = source_high / float(destination_high)
    destination_img = np.zeros((destination_high, destination_width, source_channel), dtype=np.uint8)
    for channel in range(3):
        for destination_y in range(destination_high):
            for destination_x in range(destination_width):
                source_x = (destination_x + 0.5) * scale_x - 0.5
                source_y = (destination_y + 0.5) * scale_y - 0.5

                source_x_0 = int(np.floor(source_x))
                source_y_0 = int(np.floor(source_y))

                source_x_1 = min(source_x_0 + 1, source_width - 1)
                source_y_1 = min(source_y_0 + 1, source_high - 1)

                interpolation_r1 = interpolate_values(source_x_0, source_x_1, source_x,
                                                      img[source_y_0, source_x_0, channel],
                                                      img[source_y_0, source_x_1, channel])
                interpolation_r2 = interpolate_values(source_x_0, source_x_1, source_x,
                                                      img[source_y_1, source_x_0, channel],
                                                      img[source_y_1, source_x_1, channel])
                destination_img[destination_y, destination_x, channel] = int(interpolate_values(source_y_0,
                                                                                                source_y_1,
                                                                                                source_y,
                                                                                                interpolation_r1,
                                                                                                interpolation_r2))
    return destination_img
