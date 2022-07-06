#!/usr/bin/env python
# -*-coding:utf-8-*-

import cv2

'''SIFT关键点'''

img = cv2.imread("lenna.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

keypoints, descriptor = cv2.xfeatures2d.SIFT_create().detectAndCompute(img, None)
keypoints_gray, descriptor_gray = cv2.xfeatures2d.SIFT_create().detectAndCompute(img_gray, None)

img = cv2.drawKeypoints(image=img, keypoints=keypoints, outImage=img, color=(0, 255, 0),
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_gray = cv2.drawKeypoints(image=img_gray, keypoints=keypoints_gray, outImage=img_gray, color=(0, 255, 0),
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("sift_keypoints", img)
cv2.imshow("sift_keypoints_gray", img_gray)
cv2.waitKey()
cv2.destroyAllWindows()
