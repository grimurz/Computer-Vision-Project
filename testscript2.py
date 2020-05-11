#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
the script is used to test the following functions 

- getRansacHomography
- warpImage

against the in-build functions from OpenCV 

- findHomography
- warpPerspective

and a simple blending test
"""

import numpy as np
import cv2
import os
cv2.ocl.setUseOpenCL(False)
import matplotlib.pyplot as plt
from skimage import util, transform

# import own functions to test
from r_homography import getPointsFromHomogeneousCoor, getRansacHomography
from warp import warpImageBasic, warpImage

def showImage(im, h, w):
    plt.figure(figsize = (h,w))
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.show 

#%% ===================== INITIAL IMAGE LOADING ===============================
# read images from path
# chance path and extention according to the images and type of images you want to test 
images = []
path = "testimages1"

for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext != ".JPG":
        continue
    images.append(cv2.imread(os.path.join(path,f)))
    
image1 = images[1]
image2 = images[0]

showImage(image1, 8, 8)
showImage(image2, 8, 8)

#%% ===================== RESCALE OF IMAGES ===================================
# =============================================================================
# """
# if images are LARGE then rescale to make a quicker test
# - used for the drone images, else uncomment the two lines!
# """
# 
# image1 = util.img_as_ubyte(transform.rescale(image1, 0.25))
# image2 = util.img_as_ubyte(transform.rescale(image2, 0.25))
# 
# showImage(image1, 8, 8)
# =============================================================================


#%% ================ POINT COORESPONDENCES USING SIFT =========================
# find initial point coorespondences - since this is in the test phase, we just 
# extract the coorespondences using in-build functions 
sift_values = cv2.xfeatures2d.SIFT_create() 
keypoints0, descriptors0 = sift_values.detectAndCompute(image1, None)
keypoints1, descriptors1 = sift_values.detectAndCompute(image2, None)
knnmatcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck = False)
raw = knnmatcher.knnMatch(descriptors0, descriptors1, 2)
matches = []
for m,n in raw:
    if m.distance < n.distance * 0.5:
        matches.append(m)
src_pts = np.float32([keypoints0[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

images_matches = cv2.drawMatches(image1,keypoints0,image2,keypoints1,matches, None, flags=2)
showImage(images_matches, 15, 10)

#%% =============== TEST OF RANSAC HOMOGRAPHY ESTIMATION ======================

# Our own function   
H1, no_inliers = getRansacHomography(src_pts, dst_pts, 5.0)
print(H1)

# The inbuild function from openCV
H2, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
print(H2)

# Prints the estimated points for H1 and H2
print("Estimated points using the two different H functions.")
print("x1,   x2 \t y1,   y2 \t\t src_pts (x,y)")
for i in range (len(src_pts)):
    a = np.asarray(src_pts[i][0])
    b = np.ones(1)
    p1 = getPointsFromHomogeneousCoor(np.dot(H1,np.concatenate((a, b), axis=None)))
    p2 = getPointsFromHomogeneousCoor(np.dot(H2,np.concatenate((a, b), axis=None)))
    # print x1, x2 then y1, y2 to be able to cpmpare 
    print(np.round(p1[0],2), np.round(p2[0],2),"\t", np.round(p1[1],2), np.round(p2[1],2), "\t", dst_pts[i][0])
    
"""adjust to fit the imagesize!"""    
h = 2000
w = 2000

# Shows the results of the two different homographies 
result1 = cv2.warpPerspective(image1, H1, (w,h))
#result1[0:image2.shape[0], 0:image2.shape[1]] = image2
showImage(result1, 8, 8)

result2 = cv2.warpPerspective(image1, H2, (w,h))
#result2[0:image2.shape[0], 0:image2.shape[1]] = image2
showImage(result2, 8, 8)

# =============================================================================
# # 
# out_image = np.zeros((h,w), object)
# #plt.imshow(out_image)
# 
# for i in range (out_image.shape[0]):
#     for j in range (out_image.shape[1]):
#         out_image[i][j] = np.array([0,0,0], dtype=np.uint8)
# 
# 
# out_image[0:h, 0:w] = np.array([0,0,0], dtype=np.uint8)
# 
# 
# out_image = np.zeros((h,w,3), 'uint8')
# out_image[..., 0] = 0
# out_image[..., 1] = 0
# out_image[..., 2] = 0
# 
# temp = np.dot(H1,[1,1,1])
# temp = getPointsFromHomogeneousCoor(temp)
# x = int(np.round(temp[0]))
# y = int(np.round(temp[1]))
# im_lenx = images[0].shape[1]
# im_leny = images[0].shape[0]
# 
# if x < 0:
#     out_image[y:im_leny+y, 0:im_lenx+x ] = images[0][0:im_leny , 0-x:im_lenx]
# elif y < 0:
#     out_image[0:im_leny+y , x:im_lenx+x ] = images[0][0-y:im_leny , 0:im_lenx]
# else:
#     out_image[y:im_leny+y, x:im_leny+x] = images[0]
# 
# 
# plt.figure
# plt.imshow(out_image)
# plt.show
# 
# =============================================================================


#%% =============== TEST OF WARPING PERSPECTIVE FUNCTION ======================

# Our own warping function - backward mapping
warped_image1 = warpImage(image1, H1, h,w)
warped_image1[0:image2.shape[0], 0:image2.shape[1]] = image2
showImage(warped_image1, 8, 8)

# Inbuild warping function + inbuild homography
warped_image2 = cv2.warpPerspective(image1, H2, (w,h))
warped_image2[0:image2.shape[0], 0:image2.shape[1]] = image2
showImage(warped_image2, 8, 8)


#%% ================= TEST OF BASIC BLENDING FUNCTION =========================
""" This is just a simple test of how well the images is suited for blending. 
The paper suggests a multi band blendning in the end of the algorithm, and we 
wanted to se if the images could be used for that. In this case we have just 
used a simple blend of two equal sized images on top of each oter using the 
cv2.addWeighted() function """

def simpleBlendTest(im1, im2, H, h, w):

    fullim = warpImage(im1, H, h, w)
    fullim[0:im2.shape[0], 0:im2.shape[1]] = im2
    
    warped = warpImage(im1, H, h, w)
    
    im = np.zeros((h,w,3),'uint8')
    im[0:im2.shape[0], 0:im2.shape[1]] = im2
    
    mask = np.any(warped != [0, 0, 0], axis=-1)  
    
    warped_op = np.where(mask[...,None]==0, im, warped)
    warped_op = np.asarray(warped_op, dtype=np.uint8)
    
    alpha = 0.5
    beta = 1 - alpha
    
    result = cv2.addWeighted(warped_op, alpha, fullim, beta, 0.0, warped_op)

    return result 


blended_image = simpleBlendTest(image1, image2, H1, h, w)
showImage(blended_image, 18, 18)

