
import numpy as np
import cv2
from matplotlib import pyplot as plt

# img1 = cv2.imread('A.jpg',0)
# img2 = cv2.imread('B.jpg',0)
img1 = cv2.imread('m1.png',0)
img2 = cv2.imread('m2.png',0)


# I. Keypoints (ORB)
orb = cv2.ORB_create(nfeatures=100)
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)


# II. Feature mathcing (Brute Force Matching)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
raw = sorted(matches, key = lambda x:x.distance)


# III. For each image:

# (i) Select m candidate matching images that have the most feature
#     matches to this image
matches = []
for m in raw:
    if m.distance < 45:     # replace later with image being compared?
        matches.append(m)


        
# (ii) Find geometrically consistent feature matches using RANSAC to
#      solve for the homography between pairs of images
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)


# (iii) Verify image matches using a probabilistic model
# ???


# IV. Find connected components of image matches
# ???


# V. For each connected component:
    # (i) Perform bundle adjustment to solve for the rotation θ1, θ2, θ3 and
    #     focal length f of all cameras

    # (ii) Render panorama using multi-band blending




##### Visuals and test code #####

h = 500
w = 500

result = cv2.warpPerspective(img1, H, (w,h))
result[0:img2.shape[0], 0:img2.shape[1]] = img2

plt.figure
plt.imshow(result),plt.show()





img_out1 = cv2.drawKeypoints(img1, kp1, outImage = None, color=(255,0,0))
img_out2 = cv2.drawKeypoints(img2, kp2, outImage = None, color=(255,0,0))

plt.imshow(img_out1),plt.show()
plt.imshow(img_out2),plt.show()



# # matching_result = cv2.drawMatches(img1, kp1, img2, kp2, raw[:20], None, flags=2)
# matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)

# cv2.imshow("Close me by pressing the any key", matching_result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





