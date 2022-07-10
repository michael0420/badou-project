import cv2

#均值哈希算法
def aHash(img):
    #缩放为8*8
    img=cv2.resize(img, (8, 8), interpolation = cv2.INTER_CUBIC)
    imgm = img.mean()
    imglist = ['1' if x>imgm else '0' for x in list(img.reshape(1,-1)[0])]
    return ''.join(imglist)


def dhash(img):
    img=cv2.resize(img, (9, 8), interpolation = cv2.INTER_CUBIC) # 8*9
    re = []
    for i in range(8):
        for j in range(8):
            if img[i,j]>img[i,j+1]:
                re.append('1')
            else:
                re.append('0')
    return ''.join(re)



img = cv2.imread('lenna.png',0)

print(dhash(img))
