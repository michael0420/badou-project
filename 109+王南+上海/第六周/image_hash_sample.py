import numpy as np
import cv2


def aHash(image):
    image88 = cv2.resize(image, (8, 8))
    gray88 = cv2.cvtColor(image88, cv2.COLOR_BGR2GRAY)
    gray88_avg = np.average(gray88)
    mask = gray88 > gray88_avg
    # print("mask:", mask)
    hash_metrics = np.zeros(gray88.shape, dtype=gray88.dtype)
    hash_metrics[mask] = 1
    # print("hash_metrics:", hash_metrics)
    return "".join(str(i) for i in hash_metrics.ravel())


def dHash(image):
    image89 = cv2.resize(image, (9, 8))
    gray89 = cv2.cvtColor(image89, cv2.COLOR_BGR2GRAY)
    mask = gray89[:, 0:8] > gray89[:, 1:9]
    hash_metrics = np.zeros(mask.shape, dtype=gray89.dtype)
    hash_metrics[mask] = 1
    return "".join(str(i) for i in hash_metrics.ravel())


def hamming_distance(str1, str2):
    length_difference = 0
    len1 = len(str1)
    len2 = len(str2)
    distance = 0

    if len1 != len2:
        length_difference = abs(len1 - len2)
        if(len1 > len2):
            str1 = str1[0:len2]
        else:
            str2 = str2[0:len1]

    for i in range(len(str1)):
        # print(f"str1: {str1[i]}, ord1:{ord(str1[i])}")
        # print(f"str2: {str2[i]}, ord2:{ord(str2[i])}")
        # print(f"^:{ ord(str1[i]) ^ ord(str2[i])}")
        if ord(str1[i]) ^ ord(str2[i]) != 0:
            distance += 1
    
    return distance + length_difference


if __name__ == "__main__":
    lena = cv2.imread("imgs/lena.jpeg")
    print(aHash(lena))
    print(dHash(lena))
    # print(hamming_distance("您好", "你好"))
    print(hamming_distance("00110011", "01001101"))
    # print(hamming_distance("abcde", "abcdx"))