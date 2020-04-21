
import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('simple2.jpg',0)
# img = cv2.imread('test.png',0)
# img = cv2.imread('Atest.jpg',0)
# img = cv2.imread('B.jpg',0)


## ERROR
#orb = cv2.ORB()

## FIX
orb = cv2.ORB_create(nfeatures=25, nlevels=16)

# find the keypoints with ORB
kp = orb.detect(img,None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)

## ERROR
#img2 = cv2.drawKeypoints(img,kp,color=(0,255,0), flags=0)

## Use This or the one below, One at a time
img2 = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# img2 = cv2.drawKeypoints(img, kp, outImage = None, color=(255,0,0))

plt.imshow(img2),plt.show()