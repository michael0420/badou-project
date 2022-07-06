import numpy as np
import cv2

#生成透视变换矩阵
def get_wrapmatrix(src,dst):
    A = [[src[0][0], src[0][1], 1, 0, 0, 0, -1 * src[0][0] * dst[0][0], -1 * src[0][1] * dst[0][0]],
        [0, 0, 0, src[0][0], src[0][1], 1, -1 * src[0][0] * dst[0][1], -1 * src[0][1]* dst[0][1]],
        [src[1][0], src[1][1], 1, 0, 0, 0, -1 * src[1][0] * dst[1][0], -1 * src[1][1] * dst[1][0]],
        [0, 0, 0, src[1][0], src[1][1], 1, -1 * src[1][0] * dst[1][1], -1 * src[1][1]* dst[1][1]],
        [src[2][0], src[2][1], 1, 0, 0, 0, -1 * src[2][0] * dst[2][0], -1 * src[2][1] * dst[2][0]],
        [0, 0, 0, src[2][0], src[2][1], 1, -1 * src[2][0] * dst[2][1], -1 * src[2][1]* dst[2][1]],
        [src[3][0], src[3][1], 1, 0, 0, 0, -1 * src[3][0] * dst[3][0], -1 * src[3][1] * dst[3][0]],
        [0, 0, 0, src[3][0], src[3][1], 1, -1 * src[3][0] * dst[3][1], -1 * src[3][1]* dst[3][1]]]
    A = np.array(A)
    A = A.reshape(8,8)
    B = np.array([dst[0][0],dst[0][1],dst[1][0],dst[1][1],dst[2][0],dst[2][1],dst[3][0],dst[3][1]]).reshape(8,1)
    X = np.linalg.inv(A).dot(B)
    X = np.append(X,1).reshape(3,3)
    return X
#根据透视变换矩阵进行图像变换
def warpPerspective(img,warpMatrix,img_new_scala):
    warpMatrix_I = np.linalg.inv(warpMatrix)
    w,h = img_new_scala
    img_new = np.zeros([img_new_scala[1],img_new_scala[0],3],img.dtype)
    for k in range(3):
        for i in range(h):
            for j in range(w):
                matrix = np.array([j,i,1]).reshape(3,1)
                p = np.dot(warpMatrix_I,matrix).reshape(3)
                px = int(p[0]/p[2])
                py = int(p[1]/p[2])
                img_new[i,j,k] = img[py,px,k]
    return img_new

print('warpMatrix')
# src = [[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]]
# src = np.array(src)
#
# dst = [[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]]
# dst = np.array(dst)
img = cv2.imread('photo1.jpg')

result3 = img.copy()


src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

warpMatrix = get_wrapmatrix(src, dst)
#使用opencv接口实现生成变换矩阵
#warpMatrix = cv2.getPerspectiveTransform(src, dst)
print(get_wrapmatrix(src,dst))
result = warpPerspective(result3, warpMatrix, (337,488))
#使用opencv接口实现透视变换
# result = cv2.warpPerspective(result3, warpMatrix, (337,488))
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)
print(result)


