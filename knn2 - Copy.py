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

# import own functions to test
from r_homography import getPointsFromHomogeneousCoor, getRansacHomography
from warp import warpImageBasic, warpImage


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
# path = "testimages5"
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
bestMatchCountList = []
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
    bestMatchingImagesCount = []
    for i in range(0, 4):
        bestMatch = np.argmax(bestMatchCount)
        
        if (bestMatchCount[bestMatch] < 1):
            break

        bestMatchingImages.append(bestMatch)
        bestMatchingImagesCount.append(bestMatchCount[bestMatch])
        print("  adding ", bestMatch, bestMatchCount[bestMatch])
        bestMatchCount[bestMatch] = -1

    bestMatchList.append(bestMatchingImages)
    bestMatchCountList.append(bestMatchingImagesCount)


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
##### Testing https://stackoverflow.com/a/24564574/2083242 #####
# https://stackoverflow.com/questions/42396860/inverse-homography

# Final image homographies initialized (one for each image)
id_m = np.identity(3)
H_f = np.repeat(id_m[:, :, np.newaxis], len(images), axis=2)

# Keep track of which images are done
im_done = [False] * len(images)
 
# Randomly select first image
im_no = 0 # np.random.randint(len(images))

anchor = images[im_no]

im_done[im_no] = True


sn, sn_m = 0, 1000
while False in im_done and sn < sn_m:
    
    # Get all done images
    im_all_done = np.where(im_done)[0]
    
    # Find a done image with most matches with a non-done image
    max_matches = 0
    
    for i in im_all_done:
        for j, m in enumerate(bestMatchList[i]):
            if im_done[m] is False and bestMatchCountList[m][j] > max_matches:
                max_matches = bestMatchCountList[m][j]
                im_no = i
    
    # Use image to find homography
    
    
    # Get best matched image
    H_inv = None
    for i, m in enumerate(bestMatchList[im_no]):

        if im_done[m] is False and im_no in bestMatchList[m]:
            
            # H_temp = H_all[im_no][i]
            H_inv = H_all[m][ bestMatchList[m].index(im_no) ] # inverted H -> Hij becomes Hji
            
            im_done[m] = True
            break
    
    if H_inv is not None:
        H_f[:,:,m] = H_f[:,:,im_no].dot(H_inv)
        
        print(H_inv)
  
    
    sn += 1 # safety net 


if sn == sn_m:
    print('Shit\'s F-ed, yo!')
else:
    print('I AM COMPLETE!!!')


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




#%% Get outer boundaries

min_x, max_x, min_y, max_y = 0,0,0,0

for i in range(H_f.shape[2]):
    x = H_f[:,:,i][0][2]    
    y = H_f[:,:,i][1][2]

    if x < min_x:
        min_x = x
        
    if x > max_x:
        max_x = x
        
    if y < min_y:
        min_y = y
        
    if y > max_y:
        max_y = y

# Init canvas
c_w = int(abs(min_x) + max_x + anchor.shape[1]*1.0)
c_h = int(abs(min_y) + max_y + anchor.shape[0]*1.0)

x_pad = int(abs(min_x))
y_pad = int(abs(min_y))

canvas = np.zeros((c_h, c_w, 3)).astype(int)


for i, im in enumerate(images):
        
    w_im = warpImage(im, H_f[:,:,i], c_h,c_w)
    
    for x in range(c_w):
        for y in range(c_h):
            
            r = w_im[:,:,0][y][x]
            g = w_im[:,:,1][y][x]
            b = w_im[:,:,2][y][x]
            
            if r!=0 or g!=0 or b!=0:
                canvas[:,:,0][y+y_pad][x+x_pad] = r
                canvas[:,:,1][y+y_pad][x+x_pad] = g
                canvas[:,:,2][y+y_pad][x+x_pad] = b


canvas[0+y_pad:anchor.shape[0]+y_pad, 0+x_pad:anchor.shape[1]+x_pad] = anchor


plt.figure()
plt.imshow(canvas)

plt.figure()
plt.imshow(cv2.imread('mountain.png'))


