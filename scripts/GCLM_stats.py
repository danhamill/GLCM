# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 14:55:46 2016

@author: dan
"""
from skimage.feature import greycomatrix,greycoprops
import numpy as np


image = np.array([[0, 0, 1, 1],[0, 0, 1, 1],[0, 2, 2, 2],[2, 2, 3, 3]], dtype=np.uint8)
    
level = 4
P = greycomatrix(image, [1], [0], levels=level, symmetric=True,normed=True)    


def mean_var(P):
    (num_level, num_level2, num_dist, num_angle) = P.shape
    I, J = np.ogrid[0:num_level, 0:num_level]
    I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
    mean_i = np.apply_over_axes(np.sum, (I * P), axes=(0, 1))[0, 0]
    diff_i = I - np.apply_over_axes(np.sum, (I * P), axes=(0, 1))[0, 0]
    var_h = np.apply_over_axes(np.sum, (P * (diff_i) ** 2), axes=(0, 1))[0, 0]
    return mean_i, var_h 

