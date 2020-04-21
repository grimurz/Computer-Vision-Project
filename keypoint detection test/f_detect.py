
'''
1. Detect features with FAST in different scales
2. Find rotation angle 𝜃 of each feature
3. Patch-test using precomputed test rotated with 𝜃 to get binary
   feature vector 𝒇
4. Feature distance 𝐷(𝒇1, 𝒇2) could be computed using, for example,
   the Hamming distance i.e. number of different bits.
   
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_fast/py_fast.html
http://www.edwardrosten.com/work/fast.html
https://github.com/deepanshut041/feature-detection


FAST:
    1. get circle ✔
    2. check contiguous ✔
    3. get and bin rotation
    4. plot on image
    
    problems:
    1. runs like shit, optimize?
    2. implementation of scaling unclear
    
    
BRIEF:
    


'''

import numpy as np
import cv2
from matplotlib import pyplot as plt
import itertools


img = cv2.imread('simple2.jpg',0)

t = 10 # threshold
n = 11#12 # number of contiguous pixels in the circle that are brighter/darker

h,w = img.shape

# Get the pixels around p
def getCirclePixels(img,y,x):
    
    cp = np.zeros(16)
    
    cp[0] = img[y-3][x]
    cp[1] = img[y-3][x+1]
    cp[2] = img[y-2][x+2]
    cp[3] = img[y-1][x+3]
    cp[4] = img[y][x+3]
    cp[5] = img[y+1][x+3]
    cp[6] = img[y+2][x+2]
    cp[7] = img[y+3][x+1]
    cp[8] = img[y+3][x]
    cp[9] = img[y+3][x-1]
    cp[10] = img[y+2][x-2]
    cp[11] = img[y+1][x-3]
    cp[12] = img[y][x-3]
    cp[13] = img[y-1][x-3]
    cp[14] = img[y-2][x-2]
    cp[15] = img[y-3][x-1]
    
    return cp.astype(int)


# Check if contiguous pixels are present
def contiguous(arr,n):
    
    d = [ sum( 1 for _ in group ) for key, group in itertools.groupby( np.concatenate((arr,arr)) == -1 ) if key ]
    b = [ sum( 1 for _ in group ) for key, group in itertools.groupby( np.concatenate((arr,arr)) == 1 ) if key ]
        
    return n in d or n in b



img2 = img
beep = []  
      
        
for y in np.arange(3,h-3):
    for x in np.arange(3,w-3):
        
        p = img[y][x]
        cp = getCirclePixels(img,y,x)
        
        cp[ cp < p-t ] = -1
        cp[ cp > p+t ] = 1
        cp[ cp > 1 ] = 0

        if contiguous(cp,n):
            
            print("beep", x,y, cp)
            beep.append((y,x))
            img2[y][x] = 150
            img2[y-1][x-1] = 150
            img2[y+1][x+1] = 150
            img2[y-1][x+1] = 150
            img2[y+1][x-1] = 150



plt.imshow(img2),plt.show()









