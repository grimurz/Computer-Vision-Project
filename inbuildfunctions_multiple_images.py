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
path = "testimages4"
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

pairs = []
src_pts_list = []
dst_pts_list = []
Hs = []
masks = []
for i in range(0,len(images)):
    for j in range(i+1,len(images)):     
        raw = knnmatcher.knnMatch(descriptors[i], descriptors[j], 2)
        
        # III. For each image:
        # (i) Select m candidate matching images that have the most feature matches to this image
        matches = []
        for m,n in raw:
            if m.distance < n.distance * 0.7:
                matches.append(m)
        
        print(i+1,j+1, 'beep', len(matches))
       
        
        if len(matches) > 60:   # <- needs re-evaluation later on
        
            # (ii) Find geometrically consistent feature matches using RANSAC to solve for the homography between pairs of images
            src_pts = np.float32([keypoints[i][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints[j][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if H is not None:   # <- sometimes somehow None even if matches > 0
            
                src_pts_list.append(src_pts)
                dst_pts_list.append(dst_pts)
                Hs.append(H)
                masks.append(mask)
                
                pairs.append((i,j,len(matches)))
                



# (iii) Verify imagematches using a probabilistic model
    


# IV. Find connected components of image matches



# V. For each connected component:
    # (i) Perform bundle adjustment to solve for the rotation θ1, θ2, θ3 and focal length f of all cameras
    # (ii) Render panorama using multi-band blending





# Output: Panoramic image(s)
h = 900
w = 900
s = 250 # shift

# for i in range(0,len(images)):
for i in range(0,len(pairs)):
    
    Hm = Hs[i]
    Hm[0][2] += s
    Hm[1][2] += s
    
    # warpedImage = cv2.warpPerspective(images[pairs[i][0]], Hs[i], (w,h))
    warpedImage = cv2.warpPerspective(images[pairs[i][0]], Hm, (w,h))
    warpedImage[0+s:images[pairs[i][1]].shape[0]+s, 0+s:images[pairs[i][1]].shape[1]+s] = images[pairs[i][1]]
    
    plt.figure(i)
    plt.title(str(pairs[i][0]+1) +' '+ str(pairs[i][1]+1))
    plt.imshow(warpedImage)
    plt.show
    







##### Testing https://stackoverflow.com/a/24564574/2083242 #####

# Initial homographies (one for each image)
id_m = np.identity(3)
H_init = np.repeat(id_m[:, :, np.newaxis], len(images), axis=2)

# Randomly select image (Well Yes, But Actually No)
im_no_init = 6 #np.random.randint(len(images))

# Find homography indexes corresponding to image
H_id = []
for i, p in enumerate(pairs):
    if p[0] == im_no_init or p[1] == im_no_init:
        H_id.append(i)

