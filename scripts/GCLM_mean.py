# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 14:55:46 2016

@author: dan
"""
from skimage.feature import greycomatrix, greycoprops
import numpy as np


image = np.array([[0, 0, 1, 1],[0, 0, 1, 1],[0, 2, 2, 2],[2, 2, 3, 3]], dtype=np.uint8)
    
    
result = greycomatrix(image, [1], [0, np.pi/2], levels=4, symmetric=True,normed=True)    


glcm_mean = np.sum(np.multiply(np.sum(result[:,:,0,1], axis=0),np.arange(np.shape(result[:,:,0,1])[0])))