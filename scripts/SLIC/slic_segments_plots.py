# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 09:55:15 2017

@author: dan
"""

from osgeo import gdal
import numpy as np
import os
from skimage.segmentation import slic, mark_boundaries
import matplotlib.pyplot as plt


def rescale(dat,mn,mx):
   """
   rescales an input dat between mn and mx
   """
   m = np.min(dat.flatten())
   M = np.max(dat.flatten())
   return (mx-mn)*(dat-m)/(M-m)+mn
   
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
if __name__ == '__main__':
    
    #Input sidescan rasters
    ss_dict = {'R01346':r"C:\workspace\Merged_SS\window_analysis\10_percent_shift\raster\ss_50_rasterclipped.tif",
                'R01765':r"C:\workspace\Merged_SS\raster\2014_09\ss_2014_09_R01765_raster.tif",
                'R01767':r"C:\workspace\Merged_SS\raster\2014_09\ss_2014_09_R01767_raster.tif"}
                             
    #Output Rasters
    ent_dict = {'R01346':r"C:\workspace\GLCM\slic_output\slic_glcm_rasters\2014_04\R01346_R01347_slic_entropy.tif",
                'R01765':r"C:\workspace\GLCM\slic_output\slic_glcm_rasters\2014_09\R01765_slic_entropy.tif",
                'R01767':r"C:\workspace\GLCM\slic_output\slic_glcm_rasters\2014_09\R01767_slic_entropy.tif"}
                
    var_dict = {'R01346':r"C:\workspace\GLCM\slic_output\slic_glcm_rasters\2014_04\R01346_R01347_slic_var.tif",
                'R01765':r"C:\workspace\GLCM\slic_output\slic_glcm_rasters\2014_09\R01765_3_slic_var.tif",
                'R01767':r"C:\workspace\GLCM\slic_output\slic_glcm_rasters\2014_09\R01767_3_slic_var.tif"}       
    
    homo_dict = {'R01346':r"C:\workspace\GLCM\slic_output\slic_glcm_rasters\2014_04\R01346_R01347_slic_homo.tif",
                 'R01765':r"C:\workspace\GLCM\slic_output\slic_glcm_rasters\2014_09\R01765_slic_homo.tif",
                 'R01767':r"C:\workspace\GLCM\slic_output\slic_glcm_rasters\2014_09\R01767_slic_homo.tif"}  
    
    shp_dict = {'R01346':r"C:\workspace\Merged_SS\window_analysis\shapefiles\tex_seg_800_3class.shp",
                'R01765':r"C:\workspace\Merged_SS\window_analysis\shapefiles\R01765.shp",
                'R01767':r"C:\workspace\Merged_SS\window_analysis\shapefiles\tex_seg_2014_09_67_3class.shp"}  
    fnames = []
    
    
    for num in [100,200,300,400,500,600,700,800,900,1000, 1500,2000]:
    #for num in [1500,2000]:
        #Create GLCM rasters, aggregrate distributions
        for (k,v), (k1,v1), (k2,v2), (k3,v3), (k4,v4) in zip(ss_dict.items(),ent_dict.items(),var_dict.items(), homo_dict.items(),shp_dict.items()):
            
            
            ss_raster = v
            ent_raster = v1
            var_raster = v2
            homo_raster= v3
            in_shp = v4
            #Find segments for GLCM calculations   
            im, xx, yy, gt = read_raster(ss_raster)
            im[np.isnan(im)] = 0
            im = rescale(im,0,1)
            
            #initialize segments for iteration
            print 'Now calculating n segmentss for slic segments for %s...' %(k,)
            segments_slic = slic(im, n_segments=int(num), compactness=0.01,slic_zero=True)        
            
            fig,ax = plt.subplots()
            ax.imshow(mark_boundaries(im, segments_slic,color=[1,0,0])) 
            title = k + ' n_segments = %s' %(str(num),)
            ax.set_title(title)
            plt.tight_layout()
            plt.savefig(r"C:\workspace\GLCM\slic_output\slic_segments_plots" + os.sep + k + "_" + str(num) + "_slic_segments.png",dpi=600)
            plt.close()