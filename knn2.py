#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#include <opencv2/core/types.hpp>
"""
Simple image stitcing

Can stitch two images together into one image in 2D.

Self-implemented KNN-part (part 2 in paper)

"""
print("Starting...")

from imutils import paths
import numpy as np
import imutils
import cv2
import operator
import os, os.path
cv2.ocl.setUseOpenCL(False)
import matplotlib.pyplot as plt

from math import sqrt
from scipy.stats import binom
from sklearn.neighbors import KDTree

np.set_printoptions(suppress=True)


def isValidMatch(nf, ni):
    
    p1 = 0.6
    p0 = 0.1
    
    p_f_m1 = binom.pmf(ni, nf, p1)
    p_f_m0 = binom.pmf(ni, nf, p0)
    
    p_m1 = 10**(-6)
    p_m0 = 1-p_m1   #is this correct???
    
    p_m1_f = (p_f_m1*p_m1)/(p_f_m0*p_m0)
    
    p_min = 0.999
    
    if p_m1_f > (1/((1/p_min)-1)):
        return True
    else:
        return False




images = []
# path = "testimages2"
#path = "testimages3"
path = "testimages4"
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext != ".png":
        continue

    print("Loading ", f)
    images.append(cv2.imread(os.path.join(path,f)))

imageCount = len(images)
print("[INFO] stitching images...")

###   Algorithm: Automatic Panorama Stitching Input: n unordered images

# I. Extract SIFT features from all n images
sift_values = cv2.xfeatures2d.SIFT_create()
# Detects keypoints and computes the descriptors
keypoints = []
descriptors = []

for i in range(0, imageCount):
    k, d = sift_values.detectAndCompute(images[i], None)
    keypoints.append(k); descriptors.append(d)

print("SIFT done")

# II. Find k nearest-neighbours for each feature using a k-d tree
#knnmatcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck = False)
#raw = knnmatcher.knnMatch(descriptors0, descriptors1, 2)

#allDescriptors = []
allDescriptors_imgIdx = []

#allDescriptors.append(descriptors[0])

#allDescriptors.append(descriptors[1])
#allDescriptors_imgIdx.extend( [ 1 ] * len(descriptors[1]) )

allDescriptors = descriptors[0]
allDescriptors_imgIdx.extend( [ 0 ] * len(descriptors[0]) )

for i in range(1, imageCount):
    allDescriptors = np.append(allDescriptors, descriptors[i], axis=0)
    allDescriptors_imgIdx.extend( [ i ] * len(descriptors[i]) )

tree = KDTree(allDescriptors)

# Iterate over images
bestMatchList = []
for img in range(0, imageCount):
    print("Best-matching img ", img)
    bestMatchCount = [ 0 ] * imageCount
    #print("first...")
    for feature in descriptors[img]:
        #print("feature: ", feature)
        dist, matching_idx = tree.query(feature.reshape(1, -1), k=4)
        for idx in matching_idx[0]:
            matchImg = allDescriptors_imgIdx[idx]
            
            if (matchImg != img):
                bestMatchCount[matchImg] += 1

    #print("Half...")

    # Find the 4 best matching images for current image
    bestMatchingImages = []
    for i in range(0, 4):
        bestMatch = np.argmax(bestMatchCount)
        
        if (bestMatchCount[bestMatch] < 1):
            break

        bestMatchingImages.append(bestMatch)
        print("  adding ", bestMatch, bestMatchCount[bestMatch])
        bestMatchCount[bestMatch] = -1

    bestMatchList.append(bestMatchingImages)


print("Best matching done")
# III. For each image:

H_all = []

# (i) Select m candidate matching images that have the most feature matches to this image
for img in range(0, imageCount):
    
    H_temp = []
    
    print("\nSecond-loop -matching img ", img)
    for imgMatch in bestMatchList[img]:
        if img == imgMatch:
            continue

        matches = []
        tree = KDTree(np.append(descriptors[imgMatch], descriptors[img], axis=0))
        
        for featureIdx in range(0, len(descriptors[img])):
            dists, indices = tree.query(descriptors[img][featureIdx].reshape(1,-1), k=2)
            
            # If match is not other image, then skip feature entirely
            if indices[0][0] >= len(descriptors[imgMatch]):
                continue
            
            if indices[0][1] >= len(descriptors[imgMatch]) or dists[0][0] < dists[0][1] * 0.7:
                matches.append( { 'featureIdx' : featureIdx, 'matchIdx' : indices[0][0] })
            
        
        # (ii) Find geometrically consistent feature matches using RANSAC to solve for the homography between pairs of images
        src_pts = np.float32([keypoints[img][m['featureIdx']].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[imgMatch][m['matchIdx']].pt for m in matches]).reshape(-1, 1, 2)

        if len(matches) > 0:
        
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            print("img, imgMatch: ", img, imgMatch, '\n', H, '\n')
            no_inliers = np.sum(mask) 
        else:
            print('No matches for', img, imgMatch)
            H = None
            no_inliers = 0
       
        # (iii) Verify imagematches using a probabilistic model
        
        if isValidMatch(len(src_pts), no_inliers):
            print("Validated match for img, imgMatch: ", img, imgMatch)
            H_temp.append(H)
        else:
            print("Not validated match for img, imgMatch: ", img, imgMatch)
            H_temp.append(None)
            #continue
        
        
        
    H_all.append(H_temp)





# IV. Find connected components of image matches

# V. For each connected component:
    # (i) Perform bundle adjustment to solve for the rotation θ1, θ2, θ3 and focal length f of all cameras
    # (ii) Render panorama using multi-band blending




#%%

# Output: Panoramic image(s)
h = 700
w = 700
s = 250 # shift

 

for i in range(len(images)):

    for j, m in enumerate(bestMatchList[i]):
        
        Hm = H_all[i][j]
        
        if Hm is not None:
            Hm[0][2] += s
            Hm[1][2] += s
        
            warpedImage = cv2.warpPerspective(images[i], Hm, (w,h))
            warpedImage[0+s:images[m].shape[0]+s, 0+s:images[m].shape[1]+s] = images[m]
            
            plt.figure()
            plt.title(str(i+1) +' '+ str(m+1) + ' ('+ str(i) +' '+ str(m) + ')' )
            plt.imshow(warpedImage)
            plt.show
  
'''
    1  2  3
    4  5  6
    7  8  9
'''

