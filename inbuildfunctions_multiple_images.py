#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple image stitcing

Can stitch multiple images together into one image in 2D.

"""

from imutils import paths
import numpy as np
import imutils
import cv2
import os, os.path
cv2.ocl.setUseOpenCL(False)
import matplotlib.pyplot as plt

# Load images    
images = []
path = "testimages2"
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext != ".png":
        continue
    images.append(cv2.imread(os.path.join(path,f)))

    
print("[INFO] stitching images...")

###   Algorithm: Automatic Panorama Stitching Input: n unordered images

# I. Extract SIFT features from all n images
sift_values = cv2.xfeatures2d.SIFT_create() 

keypoints = []
descriptors = []
for i in range(0,len(images)):
    # Detects keypoints and computes the descriptors
    keypoints_i, descriptors_i = sift_values.detectAndCompute(images[i], None)
    keypoints.append(keypoints_i)
    descriptors.append(descriptors_i)
    #keypoints1, descriptors1 = sift_values.detectAndCompute(images[1], None)


# II. Find k nearest-neighbours for each feature using a k-d tree
knnmatcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck = False)

#raw = []
pairs = []
src_pts_list = []
dst_pts_list = []
Hs = []
masks = []
for i in range(0,len(images)):          # 1-2, 1-3, 2-3
    for j in range(i+1,len(images)):     
        raw = knnmatcher.knnMatch(descriptors[i], descriptors[j], 2)
        #raw.append(raw_i)
        
        # III. For each image:
        # (i) Select m candidate matching images that have the most feature matches to this image
        matches = []
        for m,n in raw:
            if m.distance < n.distance * 0.7:
                matches.append(m)
                
        # (ii) Find geometrically consistent feature matches using RANSAC to solve for the homography between pairs of images
        src_pts = np.float32([keypoints[i][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[j][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        src_pts_list.append(src_pts)
        dst_pts_list.append(dst_pts)
        Hs.append(H)
        masks.append(mask)
        
        print('beep',i,j)
        pairs.append((i,j))
        
        # matching_result = cv2.drawMatches(images[i], kp1, images[j], kp2, matches, None, flags=2)
        
        # cv2.imshow("Close me by pressing the any key", matching_result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


# (iii) Verify imagematches using a probabilistic model
    


# IV. Find connected components of image matches

# V. For each connected component:
    # (i) Perform bundle adjustment to solve for the rotation θ1, θ2, θ3 and focal length f of all cameras
    # (ii) Render panorama using multi-band blending


# Output: Panoramic image(s)
h = 250
w = 250

for i in range(0,len(images)):
    warpedImage = cv2.warpPerspective(images[pairs[i][0]], Hs[i], (w,h))
    warpedImage[0:images[pairs[i][1]].shape[0], 0:images[pairs[i][1]].shape[1]] = images[pairs[i][1]]
    
    plt.figure(i)
    plt.title(pairs[i])
    plt.imshow(warpedImage)
    plt.show
    


# # 1 onto 2 ?
# warpedImage = cv2.warpPerspective(images[0], Hs[0], (w,h))
# warpedImage[0+2:images[1].shape[0]+2, 0+2:images[1].shape[1]+2] = images[1]
# # warpedImage[0:images[1].shape[0], 0:images[1].shape[1]] = images[1]

# plt.figure(1)
# plt.imshow(warpedImage)
# plt.show


# # 1 onto 3 ?
# warpedImage = cv2.warpPerspective(images[0], Hs[1], (w,h))
# warpedImage[0+2:images[2].shape[0]+2, 0+2:images[2].shape[1]+2] = images[2]
# # warpedImage[0:images[2].shape[0], 0:images[2].shape[1]] = images[2]

# plt.figure(3)
# plt.imshow(warpedImage)
# plt.show


# # 2 onto 3 ?
# warpedImage = cv2.warpPerspective(images[1], Hs[2], (w,h))
# warpedImage[0+5:images[2].shape[0]+5, 0+5:images[2].shape[1]+5] = images[2]
# # warpedImage[0:images[2].shape[0], 0:images[2].shape[1]] = images[2]

# plt.figure(2)
# plt.imshow(warpedImage)
# plt.show

