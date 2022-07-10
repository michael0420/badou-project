# -*- coding=utf-8 -*-

import numpy as np
import random


class KmeansCluster:

    def __init__(self, X, cluster_num, random_seed, max_iterations):
        self.X = X
        self.rows = X.shape[0]
        self.cols = X.shape[1]
        self.cluster_num = cluster_num
        self.random_seed = random_seed
        self.max_iterations = max_iterations

    def _centroids_init(self):
        random.seed(self.random_seed)
        init_centroids = self.X[random.sample(range(self.rows), self.cluster_num)]
        return init_centroids

    def _closest_centroid(self, sample, centroid):
        cluster_no = 0
        closest_distance = np.inf
        for i in range(self.cluster_num):
            distance = np.linalg.norm(sample - centroid[i])
            if distance < closest_distance:
                closest_distance = distance
                cluster_no = i
        return cluster_no

    def _calculate_centroids(self, prev_cluster):
        centroids = np.zeros((self.cluster_num, self.cols))
        for i in range(self.cluster_num):
            centroids[i] = np.mean(np.compress(prev_cluster[:, -1] == i, prev_cluster, axis=0)[:, :self.cols], axis=0)
        return centroids

    def kmeans(self):
        # 初始化类中心
        init_centroids = self._centroids_init()
        # 遍历迭代求解
        for i in range(self.max_iterations):
            # 根据当前中心点进行聚类
            cluster_label = np.apply_along_axis(lambda r: self._closest_centroid(r, init_centroids), axis=1, arr=self.X)
            data_cluster_label = np.column_stack((self.X, cluster_label))
            prev_centroids = init_centroids
            # 根据聚类结果重新计算类心
            init_centroids = self._calculate_centroids(data_cluster_label)
            diff = init_centroids - prev_centroids
            if diff.all():
                break
        sample_centroids = np.apply_along_axis(lambda r: self._closest_centroid(r, init_centroids), axis=1, arr=self.X)
        return sample_centroids, init_centroids
