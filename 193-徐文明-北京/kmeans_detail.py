import numpy as np


def oushijuli(s1,s2):
    s = 0
    for x,y in zip(s1,s2):
        s+= (x-y)*(x-y)
    return np.sqrt(s)

def kmean_detail(data,k,iter_num):
    m,n = data.shape
    cluster_centers = data[:k].copy()
    mlabel = np.array([0]*m)
    for _ in range(iter_num):
        for mi in range(m):
            minv = 1000000
            for ki in range(k):
                julitmp = oushijuli(data[mi],cluster_centers[ki])
                if julitmp<minv:
                    minv = julitmp
                    mlabel[mi] = ki
        for i in range(k):
            cluster_centers[i] = np.mean(data[mlabel==i],axis = 0)
    return mlabel





data = np.array([[0,0],[0,1],[0,5],[0,6],[7,8],[-1,2]])
print(kmean_detail(data,2,5))









