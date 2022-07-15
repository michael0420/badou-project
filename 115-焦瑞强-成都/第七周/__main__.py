# -*- coding=utf-8 -*-

import cv2
from docs.conf import img_aloeL_path, img_aloeR_path, img_lenna_path, img_box_path, img_box_in_scene
from bin.random_sample_consensus import get_random_sample_consensus
from bin.mean_diff_hash import get_mean_hash, get_diff_hash, get_hash_similarity
from bin.sift import get_sift_key_points, get_feature_match

if __name__ == "__main__":
    img_lenna = cv2.imread(img_lenna_path)
    img_aloeL = cv2.imread(img_aloeL_path)
    img_aloeR = cv2.imread(img_aloeR_path)
    img_lenna_gray = cv2.cvtColor(img_lenna, cv2.COLOR_BGR2GRAY)
    img_box = cv2.imread(img_box_path, 0)
    img_box_in_scene = cv2.imread(img_box_in_scene, 0)
    get_random_sample_consensus(2, 3, 500)
    mean_hash_similarity = get_hash_similarity(get_mean_hash(img_aloeL), get_mean_hash(img_aloeR))
    diff_hash_similarity = get_hash_similarity(get_diff_hash(img_aloeL), get_diff_hash(img_aloeR))
    print("mean hash similarity", mean_hash_similarity)
    print("diff hash similarity", diff_hash_similarity)
    get_sift_key_points(img_lenna_gray)
    get_feature_match(img_box, img_box_in_scene)



