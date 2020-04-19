
import numpy as np
import cv2
from matplotlib import pyplot as plt

# img = cv2.imread('simple.jpg',0)
# img = cv2.imread('test.png',0)

img1 = cv2.imread('A.jpg',0)
img2 = cv2.imread('B.jpg',0)
# img1 = cv2.imread('m1.png',0)
# img2 = cv2.imread('m2.png',0)


## ORB
orb = cv2.ORB_create(nfeatures=100)


# find the keypoints and compute the descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)


# Brute Force Matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)

matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)


## Use This or the one below, One at a time
# img_out1 = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# img_out2 = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_out1 = cv2.drawKeypoints(img1, kp1, outImage = None, color=(255,0,0))
img_out2 = cv2.drawKeypoints(img2, kp2, outImage = None, color=(255,0,0))

plt.imshow(img_out1),plt.show()
plt.imshow(img_out2),plt.show()

cv2.imshow("Close me by pressing the any key", matching_result)
cv2.waitKey(0)
cv2.destroyAllWindows()