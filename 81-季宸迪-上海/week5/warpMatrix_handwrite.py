# warpMatrix

import numpy as np

def WarpPerspectiveMatrix(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4, "输入，输出维度错误！"
    
    nums = src.shape[0]
    A = np.zeros((2*nums, 8)) # A*warpMatrix=B
    B = np.zeros((2*nums, 1))
    for i in range(0, nums):
        # A_i[0] = xi, A_i[1] = yi
        A_i = src[i,:]
        # A_i[0] = Xi, A_i[1] = Yi
        B_i = dst[i,:]
        A[2*i, :] = [A_i[0], A_i[1], 1, 0, 0, 0,
                       -A_i[0]*B_i[0], -A_i[1]*B_i[0]]
        B[2*i] = B_i[0]
        
        A[2*i+1, :] = [0, 0, 0, A_i[0], A_i[1], 1,
                       -A_i[0]*B_i[1], -A_i[1]*B_i[1]]
        B[2*i+1] = B_i[1]
    # change A to matrix
    A = np.mat(A)
    #用A.I求出A的逆矩阵，然后与B相乘，求出warpMatrix
    warpMatrix = A.I * B #求出[a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32].T
    # print(warpMatrix.shape) # (8,1)
    
    #之后为结果的后处理
    warpMatrix = np.array(warpMatrix).T
    # print(warpMatrix.shape) # (1,8)
    # print(warpMatrix)
    warpMatrix= warpMatrix[0]
    # print(warpMatrix.shape) # (8,)
    # print(warpMatrix)
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0) #插入a_33 = 1
    # print(warpMatrix.shape) # (9,)
    # [a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32, a_33].T to [[a_11, a_12, a_13],[a_21, a_22, a_23],[a_31, a_32, a_33]]
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix
 
if __name__ == '__main__':
    print('warpMatrix')

    src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
    dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
    
    warpMatrix = WarpPerspectiveMatrix(src, dst)
    print(warpMatrix)

