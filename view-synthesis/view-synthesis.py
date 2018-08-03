## Synthesized center view (image) using two stereo images (left and right)

import cv2
import numpy as np

#Read images and groundtruths
left_img = cv2.imread('view1.png')
right_img = cv2.imread('view5.png')
grd_disp_left = cv2.imread('disp1.png', 0)
grd_disp_right = cv2.imread('disp5.png', 0)
m, n = grd_disp_left.shape

#Create resultant image
left_img_syn = np.zeros_like(left_img)
right_img_syn = np.zeros_like(left_img)
new_img = np.zeros_like(left_img)
#Generate synthesized image from left image
for x in range(m):
    for y in range(n):
        disp = y - int(grd_disp_left[x,y]/2)
        if disp>-1 and np.all(left_img_syn[x,disp] == 0):
            left_img_syn[x,disp] = left_img[x,y]

cv2.imshow("Left" , left_img_syn)
#Generate synthesized image from right image
for x in range(m):
    for y in range(n):
        disp = y + int(grd_disp_right[x, y] / 2)
        if disp<n and np.all(right_img_syn[x,disp] == 0):
            right_img_syn[x,disp] = right_img[x,y]

cv2.imshow("Right " , right_img_syn)

#Synthesize new image in middle of left-right
for x in range(m):
    for y in range(n):
        if np.all(left_img_syn[x, y] == 0) and np.all(right_img_syn[x,y] == 0):
            new_img[x, y] = 0
        elif np.all(left_img_syn[x, y]) > np.all(right_img_syn[x,y]):
            new_img[x,y] = left_img_syn[x,y]
        else:
            new_img[x,y] = right_img_syn[x,y]

cv2.imshow("Synthesized image", new_img)

cv2.waitKey(0)
