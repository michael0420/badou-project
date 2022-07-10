# -*- coding=utf-8 -*-

import numpy as np
import cv2


def get_warp_matrix(src, dsc):
    assert src.shape[0] == dsc.shape[0] and src.shape[0] >= 4
    num = src.shape[0]
    A = np.zeros((2 * num, 8))
    B = np.zeros((2 * num, 1))
    for i in range(num):
        A_i = src[i, :]
        B_i = dsc[i, :]
        # 奇数行和偶数行
        A[i * 2, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[i * 2] = B_i[0]

        A[i * 2 + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[i * 2 + 1] = B_i[1]

    A = np.mat(A)
    warp_matrix = A.I * B
    warp_matrix = np.array(warp_matrix).T[0]
    # 插入 a33
    warp_matrix = np.insert(warp_matrix, warp_matrix.shape[0], values=1, axis=0)
    warp_matrix = warp_matrix.reshape((3, 3))
    return warp_matrix


def get_vertex(lines):
    def get_k_b(line_points):
        x_0, y_0, x_1, y_1 = line_points[0]
        k = (y_1 - y_0) / (x_1 - x_0)
        b = y_1 - k * x_1
        return k, b
    vertices = list()

    for i in range(lines.shape[0]):
        k1, b1 = get_k_b(lines[i])
        for j in range(i + 1, lines.shape[0]):
            k2, b2 = get_k_b(lines[j])
            x = (b2 - b1) / (k1 - k2)
            y = k1 * x + b1
            if x > 0 and y > 0:
                vertices.extend([(int(np.round(x, 0)), int(np.round(y)))])
    return vertices


# 计算原始顶点与目标顶点
def get_order_points(points):
    points = np.array(points)
    x_sort = np.argsort(points[:, 0])
    y_sort = np.argsort(points[:, 1])
    clock_size_point = np.array([points[x_sort[0]], points[y_sort[0]], points[x_sort[-1]], points[y_sort[-1]]],
                                dtype=np.float32)
    return clock_size_point


def get_target_point(clock_point):
    w1 = np.linalg.norm(clock_point[0] - clock_point[1])
    w2 = np.linalg.norm(clock_point[2] - clock_point[3])
    w = max([w1, w2])
    h1 = np.linalg.norm(clock_point[1] - clock_point[2])
    h2 = np.linalg.norm(clock_point[3] - clock_point[0])
    h = max([h1, h2])
    return np.array([[0, int(round(w))], [0, 0], [int(round(h)), 0], [int(round(h)), int(round(w))]])


def draw_line(img, lines):
    for lines_points in lines:
        cv2.line(img, (lines_points[0][0], lines_points[0][1]), (lines_points[0][2], lines_points[0][3]), (0, 255),
                 thickness=2, lineType=8, shift=0)
    cv2.imshow("hf", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_point(img, points):
    for p in points:
        cv2.circle(img, p, 5, (0, 0, 255), -1)
    cv2.imshow("vertex", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_img_preprocess(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_canny = cv2.Canny(img_gray, threshold1=50, threshold2=150, apertureSize=3)
    img_gaussian = cv2.GaussianBlur(img_canny, (3, 3), sigmaX=1, sigmaY=1)
    return img_gaussian
