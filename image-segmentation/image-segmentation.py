## Implementation of image segmentation using mean shift alogrithm

import cv2
import time
import numpy as np

# Meanshift parameter
h = 60  # Window size
mean_shift = 1  # Meanshift

# Finds the cluster for given centroid and windowsize in data point X
# Returns all the elements of cluster with element's index in data point
def find_cluster(X, centroid, window_size):
    eligible_x = [] # Stores cluster elements
    matching_ind = [] # Stores cluster element's index in data point
    for i in range(X.shape[0]):
        distance = np.sqrt(np.sum((centroid - X[i])**2))
        if distance <= window_size:
            eligible_x.append(X[i])
            matching_ind.append(i)
    return np.array(eligible_x), matching_ind

# Color the unlabeled neighbours with value of centroid in image
def color_unlabeled_neighbours(img, neighbours, mean):
    neighbours = neighbours[np.where(neighbours[:,5] == 0)] # get unlabeled neighbours only
    b,g,r = int(mean[0]),int(mean[1]),int(mean[2])
    for point in neighbours:
        x = point[3]-1
        y = point[4]-1
        img[x,y] = [b,g,r]
    return img



"""Start of Program"""

#Read image
ori_img = cv2.imread("Butterfly.jpg")
cv2.imshow("Original image", ori_img)
new_img = np.zeros_like(ori_img)
m, n, c = new_img.shape
b,g,r = cv2.split(ori_img)
#Create feature matrix
feature_mat = np.zeros(shape=(m*n,5), dtype=int)
row = 0
for x in range(m):
    for y in range(n):
        feature_mat[row, 0] = b[x,y]
        feature_mat[row, 1] = g[x,y]
        feature_mat[row, 2] = r[x,y]
        feature_mat[row, 3] = x+1
        feature_mat[row, 4] = y+1
        row = row+1


print("Total data points for computation",feature_mat.shape)
# Creat copy of feature matrix with sixth coloumn as mode (0 - not labeled and 1 - labeled)
X = np.insert(feature_mat, 5, 0, axis=1)

start = time.time()
segment_count = 0
unlabeled_index = (np.where(X[:, 5] == 0))[0]

# MeanShift Algorithm
while len(unlabeled_index)>0:
    print()
    ##Sample a Random point from Feature Matrix - only if mode of point is 0
    mean_index = unlabeled_index[np.random.choice(unlabeled_index.shape[0], 1, replace=False)][0]
    mean = feature_mat[mean_index] # Sample
    all_neigh_indexes = [mean_index] # the list which will contain all the neighs indexes after convergence
    print("New sample ", mean)
    while True:
        # Find the neighbours of mean within distance h
        neighbours, neigh_indexes = find_cluster(feature_mat, mean, h)
        # Calculate new mean
        new_mean = neighbours.mean(0)
        distance = np.sqrt(np.sum((mean - new_mean)**2)) # euclidean distance between mean and new mean
        # concate the neighbour indexes to final list
        all_neigh_indexes = all_neigh_indexes + neigh_indexes
        if distance <= mean_shift:
            # If new mean distance is within mean_shift then convergence reached
            print("Convergence reached with mean", new_mean)
            all_neigh_indexes = list(set(all_neigh_indexes))  # Take unique indexes only
            print("Neighbours identified", len(all_neigh_indexes))
            # Color all the unlabeled pixel of cluster with mean color in image
            new_img = color_unlabeled_neighbours(new_img, X[all_neigh_indexes,:], new_mean)
            X[all_neigh_indexes, 5] = 1  # label all the neighbours
            unlabeled_index = (np.where(X[:, 5] == 0))[0]
            print("Remaining pixels", len(unlabeled_index))
            segment_count = segment_count + 1
            break;
        #else:
        #    print("Recomputing cluster..")
        mean = new_mean


print()
print("Time taken",time.time()-start)
print("Total Segments", segment_count)

cv2.imshow("Segmented image", new_img)



cv2.waitKey(0)




