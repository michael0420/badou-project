import numpy as np


def warp_perspective_matrix(src, dst):
    assert src.shape == dst.shape and src.shape[0] >= 4

    nums = src.shape[0]
    A = np.zeros((nums*2, 8))
    B = np.zeros((nums*2, 1))
    for i in range(nums):
        A_i = src[i, :]
        B_i = dst[i, :]
        A[i*2, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0]*B_i[0], -A_i[1]*B_i[0]]
        A[i*2+1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0]*B_i[1], -A_i[1]*B_i[1]]
        B[i*2] = B_i[0]
        B[i*2+1] = B_i[1]

    A = np.mat(A)
    warp_matrix = A.I*B
    warp_matrix = np.array(warp_matrix).T[0]
    warp_matrix = np.insert(warp_matrix, warp_matrix.shape[0], values=1.0, axis=0)
    warp_matrix = warp_matrix.reshape((3, 3))

    return warp_matrix


if __name__ == '__main__':
    # 求解透视变换矩阵
    print('warpMatrix:')
    src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
    src = np.array(src)

    dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
    dst = np.array(dst)

    matrix = warp_perspective_matrix(src, dst)
    print(matrix)
