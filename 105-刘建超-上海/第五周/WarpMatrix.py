#!/usr/bin/env python
# -*-coding:utf-8-*-

import numpy as np

'''透视变换'''


def getPerspectiveTransform(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4  # 限制输入数据的数量

    nums = np.shape(src)[0]
    A = np.zeros([nums * 2, 8])
    B = np.zeros((nums * 2, 1))
    for i in range(0, nums):
        A_i = src[i, :]
        B_i = dst[i, :]
        A[i * 2, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[i * 2] = B_i[0]
        A[i * 2 + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[i * 2 + 1] = B_i[1]
    A = np.mat(A)  # 解释为矩阵
    # A*warpMatrix=B  用A.I求出A的逆矩阵，然后与B相乘，求出warpMatrix   B/A=B*(A的逆矩阵)
    warpMatrix = A.I * B
    warpMatrix = np.array(warpMatrix).T[0]
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
    warpMatrix = np.reshape(warpMatrix, (3, 3))
    # warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix


if __name__ == "__main__":
    # src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    # src = np.array(src)
    # dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
    # dst = np.array(dst)
    src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
    warpMatrix = getPerspectiveTransform(src, dst)
    print("warpMatrix:\n", warpMatrix)
