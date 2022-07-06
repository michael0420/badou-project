import numpy as np


def warpPerspectiveMatrix(src1, dst1):
    assert src1.shape[0] == dst1.shape[0] and src1.shape[0] >= 4

    nums = src1.shape[0]
    A = np.zeros((2 * nums, 8))  # A*warpMatrix=B
    B = np.zeros((2 * nums, 1))
    for i in range(0, nums):
        A_i = src1[i, :]
        B_i = dst1[i, :]
        A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0,
                       -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[2 * i] = B_i[0]

        A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1,
                           -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i + 1] = B_i[1]

    A = np.mat(A)
    # 用A.I求出A的逆矩阵，然后与B相乘，求出warpMatrix
    warpMatrix = A.I * B  # 求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32
    print(warpMatrix) # 求出8行1列矩阵，这是二维矩阵
    # 之后为结果的后处理
    warpMatrix = np.array(warpMatrix).T[0]  #  变成1维数组只有8个数,T[0]表示：8行1列矩阵转置变为1行8列矩阵，然后取第一行矩阵变成了一维数组， 注：一维数组转置还是它本身，因为它不是二维矩阵
    print(warpMatrix)
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1 axis的值：插入某一行(0)还是列(1)
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix


if __name__ == '__main__':
    print('warpMatrix')
    src = np.array([[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]])

    dst = np.array([[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]])

    warpMatrix = warpPerspectiveMatrix(src, dst)
    print(warpMatrix)
    print(src.shape[0])
