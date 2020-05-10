#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#include <opencv2/core/types.hpp>
"""
Simple image stitcing

Can stitch multiple images together into one image in 2D.

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




scaleImages = True
images = []
path = "testimages10"
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext != ".png":
        continue

    print("[INFO] Loading ", f)
    images.append(cv2.imread(os.path.join(path,f)))

imageCount = len(images)

if scaleImages:
    print('[INFO scaling images')
    for i in range(len(images)):
        scale_percent = 30 # percent of original size
        width = int(images[i].shape[1] * scale_percent / 100)
        height = int(images[i].shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        images[i] = cv2.resize(images[i], dim, interpolation = cv2.INTER_AREA) 
        #images[i] = util.img_as_ubyte(transform.rescale(images[i], 0.40)).squeeze()


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

print("[INFO] SIFT done")

# II. Find k nearest-neighbours for each feature using a k-d tree
allDescriptors_imgIdx = []

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
    
    for feature in descriptors[img]:
        dist, matching_idx = tree.query(feature.reshape(1, -1), k=4)
        for idx in matching_idx[0]:
            matchImg = allDescriptors_imgIdx[idx]
            
            if (matchImg != img):
                bestMatchCount[matchImg] += 1

    # Find the 8 best matching images for current image
    bestMatchingImages = []
    bestMatchingImagesCount = []
    for i in range(0, 8):                                   # <-- ATTN!
        bestMatch = np.argmax(bestMatchCount)
        
        if (bestMatchCount[bestMatch] < 1):
            break

        bestMatchingImages.append(bestMatch)
        bestMatchingImagesCount.append(bestMatchCount[bestMatch])
        print("  adding ", bestMatch, bestMatchCount[bestMatch])
        bestMatchCount[bestMatch] = -1

    bestMatchList.append(bestMatchingImages)
    bestMatchCountList.append(bestMatchingImagesCount)


print("[INFO] Best matching done")
# III. For each image:

H_all = []

# comparison = []


# (i) Select m candidate matching images that have the most feature matches to this image
for img in range(0, imageCount):
    
    H_temp = []
    
    print("\n Second-loop -matching img ", img)
    for imgMatch in bestMatchList[img]:
        if img == imgMatch:
            continue

        matches = []
        tree = KDTree(descriptors[imgMatch])
        
        for featureIdx in range(0, len(descriptors[img])):
            neighbors = 6
            dists, indices = tree.query(descriptors[img][featureIdx].reshape(1,-1), k=neighbors)
            
            i = 0
            while i < neighbors-1: 
                # If match is not other image, then skip feature entirely
                if indices[0][i] >= len(descriptors[imgMatch]):
                    print("debug 1")
                    continue
                                   
            
                if dists[0][i] < dists[0][i+1] * 0.5: #0.7
                #if indices[0][1] >= len(descriptors[imgMatch]) or dists[0][0] < dists[0][1] * 0.7:
                    matches.append( { 'featureIdx' : featureIdx, 'matchIdx' : indices[0][i] })
                    #print("i: ", i)
                    i = neighbors
                    #print("debug 2")
                i=i+1
                    
        
        # (ii) Find geometrically consistent feature matches using RANSAC to solve for the homography between pairs of images
        src_pts = np.float32([keypoints[img][m['featureIdx']].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[imgMatch][m['matchIdx']].pt for m in matches]).reshape(-1, 1, 2)


        # # TEST TEST
        # comparison.append( cv2.drawMatches(images[img], keypoints[img], images[imgMatch], keypoints[imgMatch], matches, None, flags=2) )



        if len(matches) > 8:           # <- ATTN!
        
            # H2, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)           # <- ATTN!
            # no_inliers = np.sum(mask) 
            H, no_inliers = getRansacHomography(src_pts, dst_pts, 5.0)
            
            print("img, imgMatch: ", img, imgMatch)
            # print(H2)
            print(H)
            
        else:
            print('No matches for', img, imgMatch)
            H = None
            no_inliers = 0
       
        # (iii) Verify imagematches using a probabilistic model
        
        # H_temp.append(H)
        
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
im_no = 6 # int(len(images)/2)  # 4 # np.random.randint(len(images))                     # <--- REMEMBER!
anchor_id = im_no
anchor = images[im_no]
im_done[im_no] = True
print('\nanchor:', im_no,'\n')


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
                break

    # Use image to find homography
    # Get best matched image
    H_inv = None
    for i, m in enumerate(bestMatchList[im_no]):

        if im_done[m] is False and im_no in bestMatchList[m]:
            
            if H_all[im_no][i] is not None:
                H_inv = H_all[m][ bestMatchList[m].index(im_no) ] # troglodyte method
                # H_inv = np.linalg.inv(H_all[im_no][i]) # better and more simple method
            else:
                H_inv = None
             
            im_done[m] = True
            break
    
    if H_inv is not None:
        H_f[:,:,m] = H_f[:,:,im_no].dot(H_inv)
        
        print(H_inv)
  
    
    sn += 1 # safety net 


if sn == sn_m:
    print('Shit\'s F-ed, yo!')
    print(im_done)
else:
    print('I AM COMPLETE!!!')


#%%

# # Output: Panoramic image(s)
# h = 650
# w = 650
# s = 0 # shift

# for i in range(len(images)):

#     print(i)
    
#     for j, m in enumerate(bestMatchList[i]):
        
#         Hm = H_all[i][j]
        
#         print(Hm)
        
#         if Hm is not None:
#             Hm[0][2] += s
#             Hm[1][2] += s
        
#             warpedImage = cv2.warpPerspective(images[i], Hm, (w,h))
#             warpedImage[0+s:images[m].shape[0]+s, 0+s:images[m].shape[1]+s] = images[m]
            
#             plt.figure()
#             plt.title(str(i) +' '+ str(m))
#             plt.imshow(warpedImage)
#             plt.show


#%% Get outer boundaries

print('\nstitching...')

min_x, max_x, min_y, max_y = 0,0,0,0

for i in range(H_f.shape[2]):
    x = H_f[:,:,i][0][2] / H_f[:,:,i][2][2]
    y = H_f[:,:,i][1][2] / H_f[:,:,i][2][2]

    if x < min_x:
        min_x = x
        
    if x > max_x:
        max_x = x
        
    if y < min_y:
        min_y = y
        
    if y > max_y:
        max_y = y
        

# # Init canvas
c_w = int(abs(min_x) + max_x + anchor.shape[1]*1.1) # images should be of roughly same size as the anchor
c_h = int(abs(min_y) + max_y + anchor.shape[0]*1.1)

# # Init canvas
# c_w = 1000 #int(abs(min_x) + max_x + anchor.shape[1]*1.3) # images should be of roughly same size as the anchor
# c_h = 1000 #int(abs(min_y) + max_y + anchor.shape[0]*1.3)

x_pad = int(abs(min_x))
y_pad = int(abs(min_y))

canvas = np.zeros((c_h, c_w, 3)).astype(int)


# [ A | anchor_id | B ] -> [ B_rev | A | anchor_id ]
# [ 1 2 3 | 4 | 5 6 7] - > [ 7 6 5 | 1 2 3 | 4 ]
rend_order = np.arange(len(images))
ro_A = rend_order[0:anchor_id]
ro_B = np.flip(rend_order[anchor_id+1:])
rend_order_nu = np.concatenate((ro_B, ro_A, [anchor_id]), axis=0)

# for i, im in enumerate(images):
for i in rend_order_nu:
    
    H_temp = H_f[:,:,i]
    H_temp[0][2] += x_pad * H_temp[2][2]
    H_temp[1][2] += y_pad * H_temp[2][2]
    
    # w_im = warpImage(im, H_temp, c_h,c_w)
    w_im = warpImage(images[i], H_temp, c_h,c_w)
    
    for x in range(c_w):
        for y in range(c_h):
            
            r = w_im[:,:,0][y][x]
            g = w_im[:,:,1][y][x]
            b = w_im[:,:,2][y][x]
            
            if r!=0 or g!=0 or b!=0:
                canvas[:,:,0][y][x] = r
                canvas[:,:,1][y][x] = g
                canvas[:,:,2][y][x] = b


# We might want to render the images in reverse order, ending on the anchor
# canvas[0+y_pad:anchor.shape[0]+y_pad, 0+x_pad:anchor.shape[1]+x_pad] = anchor

# Remove black border
gray = cv2.cvtColor(canvas.astype('uint8'),cv2.COLOR_BGR2GRAY)
_,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
_, contours, _ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
x,y,w,h = cv2.boundingRect(cnt)
crop = canvas[y:y+h,x:x+w]

# plt.figure()
# plt.imshow(canvas)

plt.figure()
plt.imshow(crop)

# plt.figure()
# plt.imshow(cv2.imread('mountain.png'))


#%%

filename = 'savedImage.jpg'
cv2.imwrite(filename, crop)

# for im in comparison:
#     plt.imshow(im)