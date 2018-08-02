## Program to identify the disparity estimate of two images using dynamic programming

import cv2
import numpy as np
import time


left_img = cv2.imread('view1.png', 0)  #read it as a grayscale image
right_img = cv2.imread('view5.png', 0)

m, n = left_img.shape
left_disp = np.zeros_like(left_img)
right_disp = np.zeros_like(right_img)

occlusioncost = 20 #(You can adjust this, depending on how much threshold you want to give for noise)

#For Dynamic Programming you have build a cost matrix. Its dimension will be numcols x numcols
cost_matrix = np.zeros(shape=(n,n), dtype=int)
direction_matrix = np.zeros(shape=(n,n), dtype=int)  #(This is important in Dynamic Programming. You need to know which direction you need traverse)

print("Disparity estimate ")
start = time.time()
for k in range(0, m):
    # We first populate the first row and column values of Cost Matrix
    for i in range(0, n):
        cost_matrix[i, 0] = i * occlusioncost
        cost_matrix[0, i] = i * occlusioncost

    # Now, its time to populate the whole Cost Matrix and DirectionMatrix
    for i in range(1, n):
        for j in range(1, n):
            diff = abs(int(left_img[k, i])- int(right_img[k, j]))
            min1 = cost_matrix[i-1, j-1] + diff
            min2 = cost_matrix[i-1, j] + occlusioncost
            min3 = cost_matrix[i,j-1] + occlusioncost
            cost_matrix[i,j] = cmin = min(min1, min2, min3)
            if min1==cmin: direction_matrix[i,j]=1
            elif min2==cmin: direction_matrix[i,j]=2
            else: direction_matrix[i,j]=3
    print("row", k)
    p=n-1
    q=n-1
    # Calculate deptmap
    while p!=0 and q!=0:
        if direction_matrix[p,q]==1:
            left_disp[k,p]= abs(p - q)
            right_disp[k,q] = abs(p - q)
            p = p-1
            q = q-1;
        elif direction_matrix[p,q]==2:
            p=p-1
        else:
            q=q-1

print("Time taken", time.time()-start)
cv2.imshow("Left Disparity", left_disp)

cv2.imshow("Right Disparity", right_disp)

cv2.waitKey(0)









