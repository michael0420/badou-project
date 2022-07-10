# -*- coding=utf-8 -*-
import cv2

from docs.conf import input_img_path
from bin.kmeans import *
from bin.gaussian_salt_pepper_noise import get_gaussian_noise, get_salt_pepper_noise

if __name__ == "__main__":
    img = cv2.imread(input_img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Kmeans
    data = img_gray.reshape((-1, 1))
    img_kmeans = KmeansCluster(X=np.float32(data), cluster_num=5, random_seed=0, max_iterations=5)
    img_labels, img_centers = img_kmeans.kmeans()

    center = np.uint8(img_centers)
    img_kmeans_cluster = center[img_labels].reshape(img_gray.shape)

    cv2.imshow('kmeans cluster', img_kmeans_cluster)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 高斯噪音
    img_gaussian = get_gaussian_noise(img_gray, ratio=0.3, sigma=0.5)
    cv2.imshow("gaussian noise", img_gaussian)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 椒盐噪音
    img_salt_pepper = get_salt_pepper_noise(img_gray, 0.2)
    cv2.imshow("salt pepper noise", img_salt_pepper)
    cv2.waitKey(0)
    cv2.destroyAllWindows()