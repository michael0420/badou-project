import numpy as np
import cv2


def warpMatrix(src,dst):
    nums = src.shape[0]
    A = np.zeros((2 * nums, 8))  # A*warpMatrix=B
    B = np.zeros((2 * nums, 1))

    for i in range(0, nums):
        A_i = src[i, :]
        B_i = dst[i, :]
        A[2 * i, :] = [A_i[0], A_i[1], 1, 0, 0, 0,
                       -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[2 * i] = B_i[0]

        A[2 * i + 1, :] = [0, 0, 0, A_i[0], A_i[1], 1,
                           -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i + 1] = B_i[1]

    A = np.mat(A)
    warpMatrix = A.I * B
    #warpMatrix = B / A # 除数 有0会出错


    warpMatrix = np.array(warpMatrix).T[0]
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix

if __name__ == '__main__':
    src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    #src = np.array(src)

    dst = np.float32([[0, 0], [377, 0], [0, 488], [377, 488]])
    #dst = np.array(dst)

    warpMatrix = warpMatrix(src, dst)
    print(warpMatrix)


    warpMatrix = np.mat(warpMatrix)
    print('warpMatrix.I',warpMatrix.I)

    #图版透视变换输出
    img = cv2.imread('photo1.jpg')
    h,w,c = img.shape

    img2 = img.copy()
    result = np.zeros((488,377,3))


    for i in range(result.shape[1]):
        for j in range(result.shape[0]):
            B =  np.matrix([i,j,1]).reshape((3,1))
            newPoint =  warpMatrix.I * B
            newx = int(np.round(newPoint[0]/newPoint[2]))
            newy = int(np.round(newPoint[1]/newPoint[2]))


            if newy >= img2.shape[0]:
                newy = img2.shape[0] - 1
            if newx >= img2.shape[1]:
                 newx = img2.shape[1] - 1

            if newy < 0:
                newy = 0
            if newx < 0:
                newx = 0

            result[j,i ] = img2[newy, newx]
            '''try:
                result[newx, newy] = img2[i, j]
            except:
                print('newy,newx', newy, '      ', newx)
                print('i,j', i, '      ', j)
            '''
    result = result.astype(np.uint8)
    cv2.imshow("result", result)
    cv2.waitKey(0)