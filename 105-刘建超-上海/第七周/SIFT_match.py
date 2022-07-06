#!/usr/bin/python
# encoding=utf-8

import cv2
import numpy as np

'''SIFT特征匹配'''


def drawKnnMatches_cv2(img1, kp1, img2, kp2, goodMatch):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1 + w2] = img2

    p1 = [kpp.queryIdx for kpp in goodMatch]
    p2 = [kpp.trainIdx for kpp in goodMatch]

    pos1 = np.int32([kp1[pp].pt for pp in p1])
    pos2 = np.int32([kp2[pp].pt for pp in p2]) + (w1, 0)

    for (x1, y1), (x2, y2) in zip(pos1, pos2):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0))

    cv2.namedWindow("match", cv2.WINDOW_NORMAL)
    cv2.imshow("match", vis)


img1 = cv2.imread("iphone1.png")
img2 = cv2.imread("iphone2.png")

sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)
matches = bf.knnMatch(des1, des2, k=2)

goodMatch = []
for m, n in matches:
    if m.distance < 0.38 * n.distance:
        goodMatch.append(m)

drawKnnMatches_cv2(img1, kp1, img2, kp2, goodMatch[:10])

cv2.waitKey(0)
cv2.destroyAllWindows()
