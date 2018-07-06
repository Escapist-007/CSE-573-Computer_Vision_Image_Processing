## Performing 2D and 1D convolution on given image
## and calculate their computation time on 100x100 filter

import numpy as np
import math
import cv2
from timeit import default_timer as timer

# Padding 0 to given width
def pad_zero(img, width, arg1, arg2):
   img[:width[0]] = 0
   img[-width[1]:] = 0
   return img

# Calculate the 2d convolution for given image
# It takes the pad_size to pad the 0 to the original image so that we don't miss any pixel while calculation
def convolve2d(img, kernel, pad_size):
    img_pad = np.lib.pad(img, pad_size, pad_zero)
    img_pad = np.array(img_pad[:, :, 1], dtype=int)
    cov_img = np.zeros_like(img)
    m,n = kernel.shape
    for x in range(img.shape[0]):  # For every pixel in the image
        for y in range(img.shape[1]):
            cov_img[x][y] = np.clip((kernel * img_pad[x:x + m, y:y + n]).sum(), 0, 255)

    return cov_img

# Calculate the 1D convolution for given image
# It takes both row and col kernel and applies them accordingly on the imgage
def convolve1d(img, row_kernel, col_kernel, pad_size):
    img_pad = np.lib.pad(img, pad_size, pad_zero)
    img_pad = np.array(img_pad[:, :, 1], dtype=int)
    cov_img = np.zeros_like(img)
    temp = np.zeros_like(img_pad)

    #Calculate the temp conv form the 1st filter
    m,n = row_kernel.shape
    for x in range(img_pad.shape[0]-pad_size):  # For every pixel in the image
        for y in range(img_pad.shape[1]-2*pad_size):
            temp[x][y] = (row_kernel * img_pad[x:x + m, y:y + n]).sum()

    #Apply second filter now
    m, n = col_kernel.shape
    for x in range(img.shape[0]):  # For every useful pixel in the temp
        for y in range(img.shape[1]):
            cov_img[x][y] = np.clip((col_kernel * temp[x:x + m, y:y + n]).sum(), 0, 255)

    return cov_img

# Calculates the final gradient (g = sqrt(gx*gx + gy*gy))
def calc_g_img(gx_img, gy_img):
    g_img = np.zeros_like(img)
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            g = math.ceil(math.sqrt(math.pow(gx_img[x, y,0], 2) + math.pow(gy_img[x, y,0], 2)))
            g_img[x][y] = np.clip(g, 0, 255)
    return g_img

"""Read image"""
img = cv2.imread('lena_gray.jpg')
cv2.imshow('original',img)


"""2D Convolution using sobel filter"""
#Kernel matrices for 2D convolution
kernel_x = np.array([[-1,0,1],
                     [-2,0,2],
                     [-1,0,1]])

kernel_y = np.array([[-1,-2,-1],
                     [0,0,0],
                     [1,2,1]])
print("2D Convolution using 3x3 Sobel filter")
start = timer()
gx_2d_img = convolve2d(img, kernel_x, 1) # 1 is the pad size for padding 0 to original image
gy_2d_img = convolve2d(img, kernel_y, 1)
g_2d_img = calc_g_img(gx_2d_img, gy_2d_img)
cv2.imshow('2D Gradient Image (Gx)',gx_2d_img)
cv2.imshow('2D Gradient Image Gy',gy_2d_img)
cv2.imshow('2D Gradient Image G',g_2d_img)
end = timer()
print("Execution time(in seconds) " , (end-start))

"""1D Convolution using sobel filter"""
#Kernel matrices for 1D convolution
kernel_col_x = np.array([[1],
                         [2],
                         [1]])
kernel_row_x = np.array([[-1,0,1]])
kernel_col_y = np.array([[-1],
                         [0],
                         [1]])
kernel_row_y = np.array([[1,2,1]])

print("1D Convolution using 3x3 Sobel filter")
start = timer()
gx_1d_img = convolve1d(img, kernel_row_x, kernel_col_x, 1)
gy_1d_img = convolve1d(img, kernel_row_y, kernel_col_y, 1)
g_1d_img = calc_g_img(gx_1d_img, gy_1d_img)
cv2.imshow('1D Gradient Image (Gx)',gx_1d_img)
cv2.imshow('1D Gradient Image Gy',gy_1d_img)
cv2.imshow('1D Gradient Image G',g_1d_img)
end = timer()
print("Execution time(in seconds) " , (end-start))

""" Comparing the images obtained from 2D and 1D convolution """
print()
print("Image difference between 2D-Gx-image and 1D-Gx-image :" , (gx_2d_img-gx_1d_img).sum())
print("Image difference between 2D-Gy-image and 1D-Gy-image :" , (gy_2d_img-gy_1d_img).sum())
print("Image difference between 2D-G-image and 1D-G-image :" , (g_2d_img-g_1d_img).sum())

"""2D Convolution using 100x100 sobel filter"""
print()
col_kernel = np.random.randint(1,10,size=(100,1))
row_kernel = np.random.randint(1,10,size=(1,100))
kernel = np.matmul(col_kernel,row_kernel)
print("Matrix size", kernel.shape)
print("2D Convolution using 100x100 Sobel filter")
start = timer()
gx_2d_100 = convolve2d(img, kernel, 50)  # 50 is the pad size
end = timer()
print("Execution time(in seconds) " , (end-start))

"""1D Convolution using 100x100 sobel filter"""
print("1D Convolution using 100x100 sobel filter")
start = timer()
gx_1d_100 = convolve1d(img, row_kernel, col_kernel, 50)  # 50 is the pad size
end = timer()
print("Execution time(in seconds) " , (end-start))

print("Image difference between 2D-Gx-image and 1D-Gx-image for 100x100 filter :" , (gx_2d_100-gx_1d_100).sum())


cv2.waitKey(0)