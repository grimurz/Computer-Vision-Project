#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 15:32:31 2020

@author: Stina
"""
import numpy as np
from r_homography import getPointsFromHomogeneousCoor
import cv2

def warpImageBasic(image, H, h, w):

    out_im = np.zeros((h,w,3), 'uint8')
    out_im[..., 0] = 0
    out_im[..., 1] = 0
    out_im[..., 2] = 0
    
    for j in range (image.shape[0]): # rows
        for i in range (image.shape[1]): # col
            q = np.dot(H,[i+1,j+1,1])
            p = getPointsFromHomogeneousCoor(q)
            x = int(np.round(p[0]))
            y = int(np.round(p[1]))
            if x < 0 or y < 0:
                continue
            else:
                out_im[y:y+1, x:x+1] = image[j:j+1,i:i+1]
                                
    return out_im



def warpImage(image, H, h, w):
    
    H_inv = np.linalg.inv(H) 
    
    x = np.arange(1,w+1)
    y = np.arange(1,h+1)
    xv, yv = np.meshgrid(x, y)
    xv = xv.flatten('C')
    yv = yv.flatten('C')
    zv = np.ones(len(xv))
       
    qi = np.vstack((xv,yv,zv))
    q = np.dot(H_inv,qi[0:len(xv), :])
    p = getPointsFromHomogeneousCoor(q[0:len(xv), :])
    
    x = p[0].astype(int)
    y = p[1].astype(int)
    x = x.reshape(h, w).astype(np.float32)
    y = y.reshape(h, w).astype(np.float32)
    
    out = cv2.remap(image, x, y, cv2.INTER_LINEAR)
                              
    return out



def warpImage2(image, H, h, w, ph, pw):
    
    H_inv = np.linalg.inv(H) 
    
    x = np.arange(1,w+1)
    y = np.arange(1,h+1)
    xv, yv = np.meshgrid(x, y)
    xv = xv.flatten('C')
    yv = yv.flatten('C')
    zv = np.ones(len(xv))
       
    qi = np.vstack((xv,yv,zv))
    q = np.dot(H_inv,qi[0:len(xv), :])
    p = getPointsFromHomogeneousCoor(q[0:len(xv), :])
    
    x = p[0].astype(int)
    y = p[1].astype(int)
    x = x.reshape(h, w).astype(np.float32)
    y = y.reshape(h, w).astype(np.float32)
    
    out = cv2.remap(image, x, y, cv2.INTER_LINEAR)
                              
    return out




