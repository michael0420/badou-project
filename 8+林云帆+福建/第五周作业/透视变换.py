import numpy as np
import cv2


def Warp(src, dst):
    nums = src.shape[0]
    A = np.zeros((2 * nums, 8))
    B = np.zeros((2 * nums, 1))

    for i in range(nums):
        A_i = src[i, :]
        B_i = dst[i, :]
        A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[2 * i] = B_i[0]
        A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i + 1] = B_i[1]

    A = np.mat(A)

    warpMatrix = A.I * B
    warpMatrix = np.array(warpMatrix).T[0]
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)
    warpMatrix = warpMatrix.reshape(3, 3)
    # print(warpMatrix)
    return warpMatrix


if __name__ == '__main__':
    src = [[207, 151], [517, 285], [17, 601], [343, 731]]
    src = np.array(src)
    dst = [[0, 0], [337, 0], [0, 488], [337, 488]]
    dst = np.array(dst)

    warpMatrix = Warp(src, dst)
    img = cv2.imread('photo1.jpg')
    result = cv2.warpPerspective(img, warpMatrix, (337, 488))
    cv2.imshow('src', img)
    cv2.imshow('dst', result)
    cv2.waitKey(0)


