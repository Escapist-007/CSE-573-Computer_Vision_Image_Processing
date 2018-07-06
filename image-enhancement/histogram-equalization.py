## Program for Histogram equalization on a less contrasting image
## and plots the various equalization statistics

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2


""" Step 1 : Read file """
image = cv2.imread("dull.jpg", 0)
old_img = np.asarray(image)
m, n = old_img.shape
# Creating new image variable
new_img = np.zeros_like(old_img)

""" Step 2 : Histogram calculation """
old_hist = [0.0] * 256
for i in range(m):
    for j in range(n):
        old_hist[old_img[i, j]]+=1
old_hist = np.array(old_hist)
#print("old hist " , old_hist)

""" Step 3 : Cumulative Distribution function """
cdf = [0.0] * 256
sum = 0.0
for i in range(len(old_hist)):
    sum += old_hist[i]
    cdf[i] = sum
cdf = np.array(cdf)
#print("cdf " , cdf)

""" Step 4 : Histogram equalization - transformation function"""
trans_func = [0.0] * 256  ## Transformation function
for i in range(0, len(cdf)):
    trans_func[i] = round(cdf[i]/(n*m) * 255)
trans_func = np.array(trans_func, dtype=int)
#print("new h " , trans_func)

""" Step 5 : Create new equalized image """
for i in range(0, m):
    for j in range(0, n):
        new_img[i,j] = trans_func[old_img[i,j]]


"""PLOT FOR THE HISTOGRAM AND IMAGE TRANSFORMATION FUNCTION"""

# original histogram
gs = gridspec.GridSpec(2, 4)
gs.update(wspace=0.5)
ax1 = plt.subplot(gs[0, :2], )
cdf_normalized = cdf * old_hist.max()/ cdf.max()
plt.plot(cdf_normalized, color = 'b', label='normalized-cdf')
plt.hist(old_img.flatten(),256,[0,256], color='r', label='histogram')
plt.legend(loc = 'upper left')
plt.title('Original image histogram')

# hist of the eqlauized img
ax2 = plt.subplot(gs[0, 2:])
plt.hist(new_img.flatten(),256,[0,256], color = 'r')
plt.title('New image histogram')

# Transformation function
ax3 = plt.subplot(gs[1, 1:3])
plt.plot(trans_func)
plt.xlabel("Original intensity")
plt.ylabel("Transformed intensity")
plt.title('Transformation function')
plt.show()

""" SHOW OLD & NEW IMAGE """
# old image
plt.subplot(121)
plt.imshow(old_img)
plt.title('Before Histogram Equalization')
plt.set_cmap('gray')
# new image
plt.subplot(122)
plt.imshow(new_img)
plt.title('After Histogram Equalization')
plt.set_cmap('gray')
plt.show()
