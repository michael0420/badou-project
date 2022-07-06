import numpy as np
import random

spots = [[0, 0],
         [1, 2],
         [3, 1],
         [8, 8],
         [9, 10],
         [10, 7],
         [3, 3],
         [11, 11]]


def euclidean_distance(spot1, spot2):
    difference = spot1-spot2
    return np.sqrt(np.sum(np.dot(difference, difference)))


def find_centroid(centroid1, centroid2, spots):
    centroid_1_list = []
    centroid_2_list = []
    for spot in spots:
        centroid_1_spot_dis = euclidean_distance(np.array(centroid1), np.array(spot))
        centroid_2_spot_dis = euclidean_distance(np.array(centroid2), np.array(spot))
        if centroid_1_spot_dis < centroid_2_spot_dis:
            centroid_1_list.append(spot)
        else:
            centroid_2_list.append(spot)
    return centroid_1_list, centroid_2_list


k = 2
centroid = random.sample(spots, 2)
print('the initialized centroid:', centroid)
centroid_1 = centroid[0]
centroid_2 = centroid[1]
centroid_1_list, centroid_2_list = find_centroid(centroid_1, centroid_2, spots)
while True:
    new_centroid_1 = np.mean(np.array(centroid_1_list))
    new_centroid_2 = np.mean(np.array(centroid_2_list))
    new_centroid_1_list, new_centroid_2_list = find_centroid(new_centroid_1, new_centroid_2, spots)

    if new_centroid_1_list == centroid_1_list and new_centroid_2_list == centroid_2_list:
        break
    else:
        centroid_1_list = new_centroid_1_list
        centroid_2_list = new_centroid_2_list

print('first cluster:\n', new_centroid_1_list)
print('second cluster:\n', new_centroid_2_list)