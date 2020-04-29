#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
the script is used to test the following functions 

- getRansacHomography
- warpImage

against the in-build functions from OpenCV 

- findHomography
- warpPerspective
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

def showImage(im, h, w):
    plt.figure(figsize = (h,w))
    plt.imshow(im)
    plt.show 

# =============================================================================
"""
if images are LARGE then rescale to make a quicker test
- used for the drone images 
"""

image1 = util.img_as_ubyte(transform.rescale(image1, 0.15))
image2 = util.img_as_ubyte(transform.rescale(image2, 0.15))

showImage(image1, 8, 8)

# =============================================================================
# find initial point coorespondences - since this is in the test phase, we just 
# extract the coorespondences using in-build functions 
sift_values = cv2.xfeatures2d.SIFT_create() 
keypoints0, descriptors0 = sift_values.detectAndCompute(image1, None)
keypoints1, descriptors1 = sift_values.detectAndCompute(image2, None)
knnmatcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck = False)
raw = knnmatcher.knnMatch(descriptors0, descriptors1, 2)
matches = []
for m,n in raw:
    if m.distance < n.distance * 0.7:
        matches.append(m)
src_pts = np.float32([keypoints0[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)


# ================ TEST OF RANSAC HOMOGRAPHY ESTIMATION =======================

# Our own function   
H1, no_inliers = getRansacHomography(src_pts, dst_pts, 5.0)
print(H1)

# The inbuild function from openCV
H2, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Prints the estimated points for H1 and H2
for i in range (len(dst_pts)):
    a = np.asarray(dst_pts[i][0])
    b = np.ones(1)
    p1 = getPointsFromHomogeneousCoor(np.dot(H1,np.concatenate((a, b), axis=None)))
    p2 = getPointsFromHomogeneousCoor(np.dot(H2,np.concatenate((a, b), axis=None)))
    # print x1, x2 then y1, y2 to be able to cpmpare 
    print(np.round(p1[0],2), np.round(p2[0],2),"\t", np.round(p1[1],2), np.round(p2[1],2))
    
    
h = 800
w = 800

# Shows the results of the two different homographies 
result1 = cv2.warpPerspective(image1, H1, (w,h))
result1[0:image2.shape[0], 0:image2.shape[1]] = image2
showImage(result1, 8, 8)

result2 = cv2.warpPerspective(image1, H2, (w,h))
result2[0:image2.shape[0], 0:image2.shape[1]] = image2
showImage(result2, 8, 8)

# =============================================================================
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
# =============================================================================

# =============================================================================
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

# ================ TEST OF WARPING PERSPECTIVE FUNCTION =======================

# Simple forward mapping - just used for test (and basic starting point)
warped_image1 = warpImageBasic(image1, H1, 800,800)
warped_image1[0:image2.shape[0], 0:image2.shape[1]] = image2
showImage(warped_image1, 8, 8)

# Our own warping function - backward mapping
warped_image2 = warpImage(image1, H1, 800,800)
warped_image2[0:image2.shape[0], 0:image2.shape[1]] = image2
showImage(warped_image2, 8, 8)

# Inbuild warping function + inbuild homography
result4 = cv2.warpPerspective(image1, H2, (800,800))
result4[0:image2.shape[0], 0:image2.shape[1]] = image2
showImage(result4, 8, 8)







