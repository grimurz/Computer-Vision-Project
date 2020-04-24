
import numpy as np
import random
import math
import cv2
from scipy.stats import bernoulli
from scipy.stats import binom




# =============================================================================
# nf = 150
# ni = 133
# #nf = N - ni
# 
# p1 = 0.6
# p0 = 0.1
# 
# p_f_m1 = binom.pmf(ni, nf, p1)
# p_f_m0 = binom.pmf(ni, nf, p0)
# 
# p_m1 = 10**(-6)
# p_m0 = ???????
#
# p_m1_f = 1/(1+(p_f_m0)*p_m0/(p_f_m1*p_m1))
# 
# p_min = 0.999
# 
# if p_m1_f > p_min:
#     print("Valid match")
# else:
#     print("Invalid match")
# 
# alpha = 8.0
# beta = 0.3
# 
# alpha + beta * nf
# =============================================================================


def getPointsFromHomogeneousCoor(q):
    
    q[0] = q[0]/q[2]
    q[1] = q[1]/q[2]
    
    return q[0:2]


def getHomography (points1, points2):
    
    B = []    
# =============================================================================
#     # first try with lecture notes    
#     for p1, p2 in zip(new_points1, new_points2): 
#         x1, y1, x2, y2, = p1[0], p1[1], p2[0], p2[1]
#     
#         B.append([[0, -x2, x2*y1, 0, -y2, y2*y1, 0, -1, y1],
#                  [x2, 0, -x2*x1, y2, 0, -y2*x1, 1, 0, -x1], 
#                  [-x2*y1, x2*x1, 0, -y2*y1, y2*x1, 0, -y1, x1, 0]])
#     
#         
#     U, s, V = np.linalg.svd(B, full_matrices=True)
#     H = V[-1][-1].reshape(3, 3) 
# =============================================================================
    
# =============================================================================
#     # second try with lecture notes
#     for p1, p2 in zip(points1, points2): 
#         x1, y1, x2, y2, = p1[0], p1[1], p2[0], p2[1]
#     
#         B.append([0, -x2, x2*y1, 0, -y2, y2*y1, 0, -1, y1])
#         B.append([x2, 0, -x2*x1, y2, 0, -y2*x1, 1, 0, -x1]) 
#         B.append([-x2*y1, x2*x1, 0, -y2*y1, y2*x1, 0, -y1, x1, 0])
#     
#         
#     U, s, V = np.linalg.svd(B, full_matrices=True)#     
#     H = V[-1].reshape(3, 3) 
# =============================================================================
    
    # third try with Google..
    for p1, p2 in zip(points1, points2): 
        x1, y1, x2, y2, = p1[0], p1[1], p2[0], p2[1]
    
        # build B matrix of b-vectors for the matching points
        B.append([0, 0, 0, -x1, -y1, -1, y2*x1, y2*y1, y2])
        B.append([x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2]) 
        
    # Solve using svd
    U, s, V = np.linalg.svd(B, full_matrices=True)
    
    H = V[-1].reshape(3, 3)     
    
    return H


def getRansacHomography(p_1, p_2):
    
    # because of wierd output from cv-function..
    points1 = []
    points2 = []
    
    for i in range (len(p_1)):
        points1.append(p_1[i][0])
        points2.append(p_2[i][0])
        
    # numbers from the paper
    n = 500
    r = 4
    
    valid_error = 150 # ? pixels ?
    homographies = []
    
    for i in range (n):
        # select four random point pairs of input points
        points = list(zip(points1, points2))  
        points = random.sample(points, r) 
        print(points)
        p1, p2 = zip(*points)
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)
        points2 = np.asarray(points2)
    
        # compute homography of the four random selected points
        H = (getHomography(p1, p2))
                
        # add 1's to get the points as homogeneous coordinates 
        q2 = np.concatenate((points2,np.ones((points2.shape[0],1))),1)
    
        # for each homography compute the number of inliers according to the 
        # maximum valid error 
        no_inliers = 0
        for i in range (len(points2)):
            # estimation of q1_i from H.q2_i
            e_p1 = np.dot(H,q2[i].T) 
            e_p1 = getPointsFromHomogeneousCoor(e_p1)

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
        homographies.append((no_inliers, H)) # evt change to just update H instead of saving all H
    
    # end loop
    # return H with lmaximum number of of inliers (max, H)
    return max(homographies,key=lambda item:item[0])




