# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 16:21:33 2017

@author: dan
"""

import numpy as np
from skimage.segmentation import slic, mark_boundaries
from osgeo import gdal

def read_raster(in_raster):
    in_raster=in_raster
    ds = gdal.Open(in_raster)
    data = ds.GetRasterBand(1).ReadAsArray()
    data[data==-99] = np.nan
    gt = ds.GetGeoTransform()
    xres = gt[1]
    yres = gt[5]
    
    # get the edge coordinates and add half the resolution 
    # to go to center coordinates
    xmin = gt[0] + xres * 0.5
    xmax = gt[0] + (xres * ds.RasterXSize) - xres * 0.5
    ymin = gt[3] + (yres * ds.RasterYSize) + yres * 0.5
    ymax = gt[3] - yres * 0.5
    del ds
    # create a grid of xy coordinates in the original projection
    xx, yy = np.mgrid[xmin:xmax+xres:xres, ymax+yres:ymin:yres]
    return data, xx, yy, gt  
    
def rescale(dat,mn,mx):
   """
   rescales an input dat between mn and mx
   """
   m = np.min(dat.flatten())
   M = np.max(dat.flatten())
   return (mx-mn)*(dat-m)/(M-m)+mn
   
def crop_toseg(mask, im):
   true_points = np.argwhere(mask)
   top_left = true_points.min(axis=0)
   # take the largest points and use them as the bottom right of your crop
   bottom_right = true_points.max(axis=0)
   return mask[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1], im[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]  


ss_dict = {'R01346':r"C:\workspace\Merged_SS\window_analysis\10_percent_shift\raster\ss_50_rasterclipped.tif",
        'R01765':r"C:\workspace\Merged_SS\raster\2014_09\ss_2014_09_R01765_raster.tif",
        'R01767':r"C:\workspace\Merged_SS\raster\2014_09\ss_2014_09_R01767_raster.tif"}


for (k,v) in ss_dict.items()[0:1]:
    im,xx,yy,gt = read_raster(v)
    im[np.isnan(im)] = 0
    im = rescale(im,0,1)
    
    #initialize segments for iteration
    print 'Now calculating n segmentss for slic segments for %s...' %(k,)
    segments_slic = slic(im, n_segments=int(900), compactness=0.01)        

    for k in np.unique(segments_slic):
        mask = np.zeros(im.shape[:2], dtype = "uint8")
        mask[segments_slic == k] = 255
        count = np.count_nonzero(mask)
        im_count =  np.count_nonzero(im[segments_slic == k])
   
        #Check to make sure GLCM calculations are only made for segments with data
        if im_count > 0.75*count:
            cmask, cim = crop_toseg(mask, im)