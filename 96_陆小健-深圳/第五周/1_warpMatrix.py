import numpy as np
import cv2
 
def WarpPerspectiveMatrix(src, dst):
    # 判断在原图和变换后的图上取点数是否一样，且大于等于4个
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4    
    nums = src.shape[0]
    A = np.zeros((2*nums, 8)) # A*warpMatrix=B
    B = np.zeros((2*nums, 1))
    for i in range(0, nums):
        A_i = src[i,:]
        B_i = dst[i,:]
        A[2*i, :] = [A_i[0], A_i[1], 1, 0, 0, 0, -A_i[0]*B_i[0], -A_i[1]*B_i[0]]
        B[2*i] = B_i[0]
        
        A[2*i+1, :] = [0, 0, 0, A_i[0], A_i[1], 1, -A_i[0]*B_i[1], -A_i[1]*B_i[1]]
        B[2*i+1] = B_i[1]
 
    A = np.mat(A)  # mat可以从字符串或列表中生成数组
    #用A.I求出A的逆矩阵，然后与B相乘，求出warpMatrix
    warpMatrix = A.I * B #求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32
    
    #之后为结果的后处理
    warpMatrix = np.array(warpMatrix).T[0]
    warpMatrix = np.insert(warpMatrix, warpMatrix.shape[0], values=1.0, axis=0) #插入a_33 = 1
    warpMatrix = warpMatrix.reshape((3, 3))
    return warpMatrix
 
if __name__ == '__main__':
    print('warpMatrix')
    src = [[207, 151], [517, 285], [17, 601], [343, 731]]
    src = np.array(src)
    
    dst = [[0, 0], [337, 0], [0, 488], [337, 488]]
    dst = np.array(dst)
    
    warpMatrix = WarpPerspectiveMatrix(src, dst)
    print(warpMatrix)
    img = cv2.imread('photo1.jpg')
    result = cv2.warpPerspective(img, warpMatrix, (337, 488))
    cv2.imshow("result", result)
    cv2.imwrite('result.jpg')
    cv2.waitKey(0)
