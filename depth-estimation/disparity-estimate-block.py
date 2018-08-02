## Program to identify the disparity estimate of two images using 3x3 and 9x9 block matching technique.
## It also checks the Mean Square Error while calculating the disparity using different blocks

import cv2
import numpy as np
import sys
import time
import math

# Padding 0 to given width
def pad_zero(img, width, arg1, arg2):
   img[:width[0]] = 0
   img[-width[1]:] = 0
   return img

left_img = cv2.imread('view1.png', 0)  #read it as a grayscale image
right_img = cv2.imread('view5.png', 0)
m, n = left_img.shape
grd_disp_left = cv2.imread('disp1.png', 0)
grd_disp_right = cv2.imread('disp5.png', 0)


# Disparity estimate of left image
def left_disp_estimate(left_img, right_img, block_size):
    start = time.time()
    # Calc how much we need to shift backward and forward from cur pixel to form a block of given block size
    bck_shift = math.floor(block_size / 2)
    fwd_shift = math.ceil(block_size / 2)
    # Pad 0 to left and right image so that we can compute disparity on complete array of image
    left_img_pad = np.array(np.lib.pad(left_img, bck_shift, pad_zero))  #create padded img from left_img
    right_img_pad = np.array(np.lib.pad(right_img, bck_shift, pad_zero))  #create padded img from right_img
    # depth map(disparity) for left
    dept_map_left = np.zeros_like(left_img)
    m, n = left_img_pad.shape
    # Calculate disparity
    for x in range(bck_shift, m - bck_shift):    # traverse left image from top to bottom
        for y1 in range(bck_shift, n - bck_shift): # traverse left image for each row from left to right
            mindist = sys.maxsize
            cur = left_img_pad[x - bck_shift:x + fwd_shift, y1 - bck_shift:y1 + fwd_shift]
            for y2 in range(y1, bck_shift-1, -1):          # traverse right image from right to left <-
                dist = np.sum((cur - right_img_pad[x - bck_shift:x + fwd_shift, y2 - bck_shift:y2 + fwd_shift]) ** 2)
                if dist < mindist:
                    mindist = dist
                    matchy = y2
            dept_map_left[x - bck_shift, y1 - bck_shift] = y1 - matchy ## y1 is left img and matchy (y2) right img
    print("Time taken", time.time() - start)
    return dept_map_left

# Disparity estimate of right image
def right_disp_estimate(left_img, right_img, block_size):
    start = time.time()
    # Calc how much we need to shift backward and forward from cur pixel to form a block of given block size
    bck_shift = math.floor(block_size/2)
    fwd_shift = math.ceil(block_size/2)
    # Pad 0 to left and right image so that we can compute disparity on complete array of image
    left_img_pad = np.array(np.lib.pad(left_img, bck_shift, pad_zero))  # create padded img from left_img
    right_img_pad = np.array(np.lib.pad(right_img, bck_shift, pad_zero))  # create padded img from right_img
    dept_map_right = np.zeros_like(right_img)  # depth map(disparity) for right image
    m, n = left_img_pad.shape
    # Calculate disparity
    for x in range(bck_shift, m - bck_shift):  # traversing the right image from top to bottom
        for y1 in range(bck_shift, n - bck_shift):  # traversing the right image from left to rifht for each row
            mindist = sys.maxsize
            cur = right_img_pad[x - bck_shift:x + fwd_shift, y1 - bck_shift:y1 + fwd_shift]
            for y2 in range(y1, n - bck_shift):  # traverse left image from left to right ->
                dist = np.sum((cur - left_img_pad[x - bck_shift:x + fwd_shift, y2 - bck_shift:y2 + fwd_shift]) ** 2)
                if dist < mindist:
                    mindist = dist
                    matchy = y2
            dept_map_right[x - bck_shift, y1 - bck_shift] = matchy - y1  ## y1 is right img and matchy (y2) left img
    print("Time taken", time.time() - start)
    return dept_map_right

def mse_for_consistency(disp_est, grd_truth):
    result = np.zeros_like(disp_est)
    for x in range(disp_est.shape[0]):
        for y in range(disp_est.shape[1]):
            if disp_est[x, y] != 0:
                result[x,y] = ((float(disp_est[x, y]) - float(grd_truth[x,y])) ** 2)
    return np.mean(result)

'''Program starts here'''


print("Disparity estimate of left image ( 3x3 block size ) ")
left_disp3x3 = left_disp_estimate(left_img, right_img, 3)
cv2.imshow("Left disparity (3x3 block)", left_disp3x3)
mse = np.sum((left_disp3x3.astype("float") - grd_disp_left.astype("float")) ** 2)/float(m*n)
print("MSE: ", mse)

print("Disparity estimate of right image ( 3x3 block size ) ")
right_disp3x3 = right_disp_estimate(left_img, right_img, 3)
cv2.imshow("Right disparity (3x3 block)", right_disp3x3)
mse = np.sum((right_disp3x3.astype("float") - grd_disp_right.astype("float")) ** 2)/float(m*n)
print("MSE: ", mse)

print("Consistency check for left disparity")
result1 = np.zeros_like(left_img)
for x in range(m):
    for y in range(n):
        if left_disp3x3[x,y] == right_disp3x3[x,y-left_disp3x3[x,y]]:
            result1[x,y] = left_disp3x3[x,y]
        else: result1[x,y] = 0
cv2.imshow("Consistency check for left disp (3x3 block) ", result1)
print("MSE", mse_for_consistency(result1, grd_disp_left))

print("Consistency check for right disparity")
result2 = np.zeros_like(left_img)
for x in range(m):
    for y in range(n):
        if right_disp3x3[x,y] == left_disp3x3[x,y+right_disp3x3[x,y]]:
            result2[x,y] = right_disp3x3[x,y]
        else: result2[x,y] = 0
cv2.imshow("Consistency check for right disp (3x3 block) ", result2)
print("MSE", mse_for_consistency(result2, grd_disp_right))

print()
print("Disparity estimate of left image ( 9x9 block size ) ")
left_disp9x9 = left_disp_estimate(left_img, right_img, 9)
cv2.imshow("Left disparity (9x9 block)", left_disp9x9)
mse = np.sum((left_disp9x9.astype("float") - grd_disp_left.astype("float")) ** 2)/float(m*n)
print("MSE: ", mse)

print("Disparity estimate of right image ( 9x9 block size ) ")
right_disp9x9 = right_disp_estimate(left_img, right_img, 9)
cv2.imshow("Right disparity (9x9 block)", right_disp9x9)
mse = np.sum((right_disp9x9.astype("float") - grd_disp_right.astype("float")) ** 2)/float(m*n)
print("MSE: ", mse)

print("Consistency check on left disparity")
result3 = np.zeros_like(left_img)
for x in range(m):
    for y in range(n):
        if left_disp9x9[x,y] == right_disp9x9[x,y-left_disp9x9[x,y]]:
            result3[x,y] = left_disp9x9[x,y]
        else: result3[x,y] = 0
cv2.imshow("Consistency check left disp (9x9 block) ", result3)
print("MSE", mse_for_consistency(result3, grd_disp_left))

print("Consistency check for right disparity")
result4 = np.zeros_like(left_img)
for x in range(m):
    for y in range(n):
        if right_disp9x9[x,y] == left_disp9x9[x,y+right_disp9x9[x,y]]:
            result4[x,y] = right_disp9x9[x,y]
        else: result4[x,y] = 0
cv2.imshow("Consistency check for right disp (9x9 block) ", result4)
print("MSE", mse_for_consistency(result4, grd_disp_right))

cv2.waitKey(0)



