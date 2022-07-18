# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 15:26:54 2022

@author: Administrator
"""

import random
import numpy as np
import matplotlib.pyplot as plt

def distance(point1,point2):
    return np.sqrt(np.sum(point1-point2) ** 2)

def k_means(data, k, max_iter=10):
    centers = {}
    n_data = data.shape[0]
    for idx, i in enumerate(random.sample(range(n_data), k)):
        centers[idx] = data[i]
    for i in range(max_iter):
        print('开始第{}次迭代'.format(i))
        clusters={}
        for j in range(k):
            clusters[j]=[]
        
        for sample in data:
            distances=[]
            for c in centers:
                distances.append(distance(sample,centers[c]))
            idx = np.argmin(distances)
            clusters[idx].append(sample)
        
        pre_centers = centers.copy()
        
        for c in clusters.keys():
            centers[c] = np.mean(clusters[c], axis=0)
        
        is_convergent = True
        for c in centers:
            if distance(pre_centers[c],centers[c] > 1e-3):
                is_convergent = False
                break
            
        if is_convergent == True:
            break
    
    return centers, clusters

def predict(p_data, centers):
    distances = [distance(p_data, centers[c]) for c in centers]
    return np.argmin(distances)

if __name__ == '__main__':
    x = np.random.randint(0, high=10, size=(200,2))
    centers, clusters = k_means(x, 3)
    
    for center in centers:
        plt.scatter(centers[center][0],centers[center][1],marker='*',s=150)
        
    colors = ['r','b','y','m','c','g']
    
    for c in clusters:
        for point in clusters[c]:
            plt.scatter(point[0],point[1],c=colors[c])