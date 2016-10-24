# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 12:09:09 2016

@author: dan
"""
import numpy as np
from skimage.feature import greycomatrix, greycoprops


   
            
image = np.array([[0, 0, 1, 1],
                    [0, 0, 1, 1],
                     [0, 2, 2, 2],
                    [2, 2, 3, 3]], dtype=np.uint8)
                    
                    
g = greycomatrix(image, [1,2,3,5], [0, np.pi/2, np.pi+np.pi/2], levels=4, normed=True, symmetric=True)

(num_level, num_level2, num_dist, num_angle) = g.shape
horizontal_entropy = np.nansum(np.nansum(np.log(g[:,:,0,0])*-1*g[:,:,0,0],axis=1),axis=0)

h2 = np.apply_over_axes(np.nansum, (np.log(g)*-1*g), axes=(0,1,2))
