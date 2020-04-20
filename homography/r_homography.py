
import numpy as np
import random
import math

points1 = [[106.111084,  70.36306 ],[164.45834 , 197.70464 ],[165.0915  , 205.35329 ],
[169.06743 , 192.49399 ],[169.76605 , 184.32355 ],[169.76605 , 184.32355 ],
[170.4064  , 196.98799 ],[171.23209 , 136.38863 ],[172.29134 , 187.72005 ],
[173.10509 , 191.1028  ],[174.48148 , 179.52829 ],[177.40616 , 197.06789 ],[178.08504 , 173.89052 ],
[178.21529 , 132.80516 ],[178.68456 , 181.47278 ],[178.68456 , 181.47278 ],[179.0209  , 168.64249 ],
[181.0688  , 127.122635],[182.50029 , 102.02374 ],[184.45374 , 158.18408 ]]

points2 = [[ 68.69905  ,  72.374626 ], [  3.4499562, 127.69346  ], [  4.0967164, 135.34518  ], 
[  8.067299 , 122.49401  ],[  8.766053 , 114.32355  ],[  9.406407 , 126.98798  ],[ 10.232084 ,  66.38863  ],
[ 11.291337 , 117.72005  ],[ 11.291337 , 117.72005  ],[ 12.105083 , 121.1028   ],[ 13.481474 , 109.52829  ],
[ 16.406466 , 127.0679   ],[ 17.08504  , 103.89052  ],[ 17.215292 ,  62.805164 ],[ 17.684557 , 111.47277  ],
[ 17.684557 , 111.47277  ],[ 18.020906 ,  98.64249  ],[ 20.06881  ,  57.122635 ],[ 21.500286 ,  32.02374  ],
[ 23.453856 ,  88.18419  ]]


def getHomography (points1, points2):
    
    B = []
    for p1, p2 in zip(points1, points2): 
        x1, y1, x2, y2, = p1[0], p1[1], p2[0], p2[1]
    
        B.append([[0, -x2, x2*y1, 0, -y2, y2*y1, 0, -1, y1],
                 [x2, 0, -x2*x1, y2, 0, -y2*x1, 1, 0, -x1], 
                 [-x2*y1, x2*x1, 0, -y2*y1, y2*x1, 0, -y1, x1, 0]])
    
        
    U, s, V = np.linalg.svd(B, full_matrices=True)
    
    H = V[-1][-1].reshape(3, 3) 
    # isnt the '[-1][-1]' a bit wierd?
    # it does it work?
    
    return H


def getRansacHomography(points1, points2):
    
    # numbers from the paper
    n = 500
    r = 4
    
    valid_error = 250 # ? pixels ?
    homographies = []
    
    for i in range (n):
        # select four random point pairs of input points
        points = list(zip(points1, points2))  
        points = random.sample(points, r) 
        p1, p2 = zip(*points)
        p1 = np.asarray(p1)
        points2 = np.asarray(points2)
    
        # compute homography of the random selected points
        h = (getHomography(p1, p2))
        
        # add 1's to get the points as homogeneous coordinates 
        hpoints2 = np.concatenate((points2,np.ones((points2.shape[0],1))),1)
    
        # for each homography compute the number of inliers according to the 
        # maximum valid error 
        no_inliers = 0
        for i in range (len(points2)):
            # estimation of p1
            e_p1 = np.dot(h,hpoints2[i].T) 
            e_p1[0] = e_p1[0]/e_p1[2]
            e_p1[1] = e_p1[1]/e_p1[2]

            # distance between estimated p1 and original p1 
            distance = math.sqrt(((points1[i][0]-e_p1[0])**2)+((points1[i][1]-e_p1[1])**2))    
            #print(distance)
            
            # if distance between e_p1 and p1 is less than the valid error
            # we count the point as an inlier 
            if distance < valid_error:
                no_inliers += 1
            else:
                continue
        # end loop
        # homographies
        homographies.append((no_inliers, h))
    
    # end loop
    # return H with lmaximum number of of inliers (max, H)
    return max(homographies,key=lambda item:item[0]) 

print(getRansacHomography(points1, points2))




