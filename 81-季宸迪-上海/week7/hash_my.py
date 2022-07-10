import cv2


# 均值哈希
def avgHash(img):
    avgHashCode = ''
    img = cv2.resize(img, (8,8), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sum = 0
    for i in range(8):
        for j in range(8):
            sum += gray[i,j]
    avg = sum/64
    for i in range(8):
        for j in range(8):
            if gray[i,j] > avg:
                avgHashCode += '1'
            else:
                avgHashCode += '0'
    return avgHashCode


# 差值哈希
def diffHash(img):
    diffHashCode = ''
    img = cv2.resize(img, (9,8), interpolation=cv2.INTER_CUBIC)
    # print(img.shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, j+1]:
                diffHashCode += '1'
            else:
                diffHashCode += '0'
    return diffHashCode

# 不同的哈希值之间的汉明距离
def HammingDistance(hash1, hash2):
    if len(hash1) != len(hash2):
        return -1
    count = 0
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            count += 1
    return count


img = cv2.imread('lenna.png')
img_noise = cv2.imread('lenna_noise.png')
avgHash_img = avgHash(img)
avgHash_img_noise = avgHash(img_noise)
diff_avg = HammingDistance(avgHash_img, avgHash_img_noise)
print('均值哈希的差异为：', diff_avg)

diffHash_img = diffHash(img)
diffHash_img_noise = diffHash(img_noise)
diff_diff = HammingDistance(diffHash_img, diffHash_img_noise)
print('差值哈希的差异为：', diff_diff)