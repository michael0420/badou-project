#!/usr/bin/env python
# -*-coding:utf-8-*-

import cv2


# 均值哈希算法
def aHash(img):
    img = cv2.resize(img, (8, 8), interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = 0  # 像素值加和初值
    hash_str = ""  # 初始化图片指纹
    for i in range(8):
        for j in range(8):
            s = s + img_gray[i, j]
    avg = s / (8 * 8)  # 求所有像素的平均值
    for i in range(8):
        for j in range(8):
            if img_gray[i, j] > avg:
                hash_str = hash_str + "1"
            else:
                hash_str = hash_str + "0"
    return hash_str


# 差值哈希算法
def dHash(img):
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hash_str = ""  # 初始化图片指纹
    for i in range(8):
        for j in range(8):
            if img_gray[i, j] > img_gray[i, j + 1]:
                hash_str = hash_str + "1"
            else:
                hash_str = hash_str + "0"
    return hash_str


# Hash值对比
def camHash(hash1, hash2):
    n = 0  # 初始化汉明距离
    if len(hash1) != len(hash2):
        return -1
    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n + 1
    return n


img1 = cv2.imread("lenna.png")
img2 = cv2.imread("lenna_noise.png")
hash1 = aHash(img1)
hash2 = aHash(img2)
n = camHash(hash1, hash2)
print(hash1)
print(hash2)
print("均值哈希算法相似度：", n)

hash1 = dHash(img1)
hash2 = dHash(img2)
n = camHash(hash1, hash2)
print(hash1)
print(hash2)
print("差值哈希算法相似度：", n)
