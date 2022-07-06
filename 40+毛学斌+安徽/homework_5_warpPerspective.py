import numpy as np
import cv2


def warpMatrix_detail(src_in, dst_in):
    """
    计算透视变换用的转化矩阵。通过计算2组有映射关系的数据，得到它们之间的转换矩阵。
    该矩阵可以用于后期的图像旋转，缩放，截图，弯曲等操作
    输入： 两种有映射关系的数据,注意至少是4个点
    输入： 两者之间的转换矩阵。
    """
    # 1、本计算公式需要至少4个点，因此首先要检查一下个数是否满足要求
    assert src_in.shape[0] == dst_in.shape[0] >= 4, '至少输入4个点的坐标，且2组数据个数相等，形式[宽，高]'
    # 2、根据ppt上的计算公式，由至少4个点8个方程计算得到8个系数
    num = src_in.shape[0]
    A_matrix = np.zeros((2*num, 8))
    B_vector = np.zeros((2*num, 1))
    for i in range(num):
        x, y = src_in[i]
        DX, DY = dst_in[i]
        A_matrix[2*i] = x, y, 1, 0, 0, 0, -x*DX, -y*DX  # 参照ppt上的计算公式
        A_matrix[2*i+1] = 0, 0, 0, x, y, 1, -x*DY, -y*DY
        B_vector[2*i] = DX
        B_vector[2*i+1] = DY
    A_matrix = np.matrix(A_matrix)  # 这里转化成矩阵方式，为了后面矩阵除法和求逆矩阵更方便
    warpMatrix_8 = A_matrix.I * B_vector  # 根据矩阵的左除，如果A*B=C则B=A左除C=A的逆矩阵*C
    # 3、上一步只计算了8个系数，增加上第9个数“1”，并转换成3*3输出矩阵
    warpMatrix = np.insert(warpMatrix_8, 8, 1, axis=0).reshape((3, 3))
    return warpMatrix


if __name__ == '__main__':

    img = cv2.imread('./photo1.jpg')
    cv2.imshow('original image', img)
    # copy = img.copy()  # 创建一个副本，可以避免对原图像的更改,其实也不会更改原矩阵
    print('初始图像的形状', img.shape)
    src = np.array([[207, 151], [517, 285], [17, 601], [343, 731]], dtype='float32')  # 注意图像是宽*高，即列*行，左上角为0
    dst = np.array([[0, 0], [337, 0], [0, 448], [337, 448]], dtype='float32')  # 数据类型要求为浮点数.这里也是列*高
    print('原始坐标为\n', src)
    print('目标坐标为\n', dst)
    # 方法一：用自己写的版本
    matrix = warpMatrix_detail(src, dst)
    print('手写透视变换转化出来的矩阵：\n', matrix)
    img_new = cv2.warpPerspective(img, matrix, (337, 448))

    # 方法二：直接调用cv2.接口
    matrix2 = cv2.getPerspectiveTransform(src, dst)  # 通过该函数得到转换矩阵
    print('cv2计算得到的转换矩阵:\n', matrix2)
    img_new2 = cv2.warpPerspective(img, matrix2, (337, 448))  # 透视变化转化（原图，转化矩阵，转化后图像尺寸描述是列*行）
    print('cv2透视变换后的图像形状', img_new2.shape)
    # print('初始图像的形状', img.shape)  # 并不改变原图像，不用copy也行
    cv2.imshow('cv2.perspective detail/cv2 auto', np.hstack((img_new, img_new2)))  # np.hstack两个数组按列合并，必须以元组输入
    cv2.waitKey()
    if cv2.waitKey == 27:  # 按ESC退出，27代表ESC键
        cv2.destroyAllWindows()
