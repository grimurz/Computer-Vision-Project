"""
Simple image stitcing

Can stitch two images together into one image in 2D.

"""

from imutils import paths
import numpy as np
import imutils
import cv2
import os, os.path
cv2.ocl.setUseOpenCL(False)
import matplotlib.pyplot as plt
    
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
# Detects keypoints and computes the descriptors
keypoints0, descriptors0 = sift_values.detectAndCompute(images[0], None)
keypoints1, descriptors1 = sift_values.detectAndCompute(images[1], None)


# II. Find k nearest-neighbours for each feature using a k-d tree
knnmatcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck = False)
raw = knnmatcher.knnMatch(descriptors0, descriptors1, 2)

# III. For each image:

# (i) Select m candidate matching images that have the most feature matches to this image
matches = []

for m,n in raw:
    if m.distance < n.distance * 0.7:
        matches.append(m)
# (ii) Find geometrically consistent feature matches using RANSAC to solve for the homography between pairs of images
src_pts = np.float32([keypoints0[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# (iii) Verify imagematches using a probabilistic model
    


# IV. Find connected components of image matches

# V. For each connected component:
    # (i) Perform bundle adjustment to solve for the rotation θ1, θ2, θ3 and focal length f of all cameras
    # (ii) Render panorama using multi-band blending


# Output: Panoramic image(s)
h = 250
w = 250

result = cv2.warpPerspective(images[0], H, (w,h))
result[0:images[1].shape[0], 0:images[1].shape[1]] = images[1]

plt.figure
plt.imshow(result)
plt.show



