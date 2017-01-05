# -*- coding: utf-8 -*-
"""
Created on Wed Jan 04 10:04:22 2017

@author: dan
"""

from rasterstats import zonal_stats
import pandas as pd
from osgeo import osr, gdal,ogr
import numpy as np

from skimage.feature import greycomatrix, greycoprops
from skimage.segmentation import slic, mark_boundaries

import matplotlib.pyplot as plt

def entropy_calc(glcm):
    horizontal_entropy = np.apply_over_axes(np.nansum,(np.log(glcm)*-1*glcm),axes=(0,1))[0,0]
    horizontal_entropy = np.asarray([[horizontal_entropy[0,0]]])
    return horizontal_entropy
 
def mean_var(P):
    (num_level, num_level2, num_dist, num_angle) = P.shape
    I, J = np.ogrid[0:num_level, 0:num_level]
    I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
    diff_i = I - np.apply_over_axes(np.sum, (I * P), axes=(0, 1))[0, 0]
    var_h = np.apply_over_axes(np.sum, (P * (diff_i) ** 2), axes=(0, 1))[0, 0]
    return var_h 

def glcm_calc(im,segments_slic):
    m = im.copy()
    im[np.isnan(im)] = 0
    #create masks for glcm variables
    ent = np.zeros(im.shape[:2], dtype = "float64")
    var = np.zeros(im.shape[:2], dtype = "float64")
    homo = np.zeros(im.shape[:2], dtype = "float64")

    for k in np.unique(segments_slic):
       mask = np.zeros(im.shape[:2], dtype = "uint8")
       mask[segments_slic == k] = 255
       cmask, cim = crop_toseg(mask, im)

       # compute GLCM using 3 distances over 4 angles
       glcm = greycomatrix(cim, [5], [0], 256, symmetric=True, normed=True)

       #populate masks for 4 glcm variables
       ent[segments_slic == k] = entropy_calc(glcm)[0, 0]
       var[segments_slic == k] = mean_var(glcm)[0,0]
       homo[segments_slic == k] = greycoprops(glcm, 'homogeneity')[0, 0]
    ent[np.isnan(m)] = np.nan  
    var[np.isnan(m)] = np.nan  
    homo[np.isnan(m)] = np.nan  
    return (ent, var, homo)
    
def read_raster(in_raster):
    in_raster=in_raster
    ds = gdal.Open(in_raster)
    data = ds.GetRasterBand(1).ReadAsArray()
    data[data<=0] = np.nan
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

def CreateRaster(xx,yy,std,gt,proj,driverName,outFile):  
    '''
    Exports data to GTiff Raster
    '''
    std = np.squeeze(std)
    std[np.isinf(std)] = -99
    driver = gdal.GetDriverByName(driverName)
    rows,cols = np.shape(std)
    ds = driver.Create( outFile, cols, rows, 1, gdal.GDT_Float32)      
    if proj is not None:  
        ds.SetProjection(proj.ExportToWkt()) 
    ds.SetGeoTransform(gt)
    ss_band = ds.GetRasterBand(1)
    ss_band.WriteArray(std)
    ss_band.SetNoDataValue(-99)
    ss_band.FlushCache()
    ss_band.ComputeStatistics(False)
    del ds  

def make_glcm_raster(ent,var,homo,v1,v2,v3):
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(26949)
    CreateRaster(xx,yy,ent,gt,proj,'GTiff',v1)
    CreateRaster(xx,yy,var,gt,proj,'GTiff',v2)
    CreateRaster(xx,yy,homo,gt,proj,'GTiff',v3)
        
    
def agg_distributions(stats,in_shp,metric):
    #Lets get get the substrate to sort lists
    ds = ogr.Open(in_shp)
    lyr = ds.GetLayer(0)
    a=[]
    for row in lyr:
        a.append(row.substrate)
    lyr.ResetReading()
    del ds

    s, g, b = [],[],[]
    n = 0
    for item in stats:
        raster_array = item['mini_raster_array'].compressed()
        substrate = a[n]
        if substrate=='sand':
            s.extend(list(raster_array))
        if substrate=='gravel':
            g.extend(list(raster_array))
        if substrate=='boulders':
            b.extend(list(raster_array))
        n+=1
    del raster_array, substrate, n, item, 

    s_df = make_df2(s,metric)
    g_df = make_df2(g,metric)
    r_df = make_df2(b,metric)
    del s,  g,  b
    return s_df,  g_df, r_df,a
    
def make_df2(x,metric):
    df = pd.DataFrame(x,columns=[metric])
    return df

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

   
if __name__ == '__main__':
    
    #Input sidescan rasters
    ss_dict = {'R01346':r"C:\workspace\Merged_SS\window_analysis\10_percent_shift\raster\ss_50_rasterclipped.tif",
                'R01765':r"C:\workspace\Merged_SS\raster\2014_09\ss_2014_09_R01765_raster.tif",
                'R01767':r"C:\workspace\Merged_SS\raster\2014_09\ss_2014_09_R01767_raster.tif"}
                             
    #Output Rasters
    ent_dict = {'R01346':r"C:\workspace\GLCM\output\slic_glcm_rasters\2014_04\R01346_R01347_slic_entropy.tif",
                'R01765':r"C:\workspace\GLCM\output\slic_glcm_rasters\2014_09\R01765_slic_entropy.tif",
                'R01767':r"C:\workspace\GLCM\output\slic_glcm_rasters\2014_09\R01767_slic_entropy.tif"}
                
    var_dict = {'R01346':r"C:\workspace\GLCM\output\slic_glcm_rasters\2014_04\R01346_R01347_slic_var.tif",
                'R01765':r"C:\workspace\GLCM\output\slic_glcm_rasters\2014_09\R01765_3_slic_var.tif",
                'R01767':r"C:\workspace\GLCM\output\slic_glcm_rasters\2014_09\R01767_3_slic_var.tif"}       
    
    homo_dict = {'R01346':r"C:\workspace\GLCM\output\slic_glcm_rasters\2014_04\R01346_R01347_slic_homo.tif",
                'R01765':r"C:\workspace\GLCM\output\slic_glcm_rasters\2014_09\R01765_slic_homo.tif",
                'R01767':r"C:\workspace\GLCM\output\slic_glcm_rasters\2014_09\R01767_slic_homo.tif"}  
    
    shp_dict = {'R01346':r"C:\workspace\Merged_SS\window_analysis\shapefiles\tex_seg_800_3class.shp",
                'R01765':r"C:\workspace\Merged_SS\window_analysis\shapefiles\R01765.shp",
                'R01767':r"C:\workspace\Merged_SS\window_analysis\shapefiles\tex_seg_2014_09_67_3class.shp"}  
    fnames = []
    
    for (k,v), (k1,v1), (k2,v2), (k3,v3), (k4,v4) in zip(ss_dict.items(),ent_dict.items(),var_dict.items(), homo_dict.items(),shp_dict.items()):
        
        print 'Now calculating GLCM metrics for %s...' %(k,)
        ss_raster = v
        ent_raster = v1
        var_raster = v2
        homo_raster= v3

        #Find segments for GLCM calculations   
        im, xx, yy, gt = read_raster(ss_raster)
        im[np.isnan(im)] = 0
        im = rescale(im,0,1)
        segments_slic = slic(im, n_segments=500, compactness=.1)
        
        im = read_raster(ss_raster)[0]
        #Calculate GLCM metrics for slic segments
        ent,var,homo = glcm_calc(read_raster(ss_raster)[0],segments_slic)
        
        print 'Now making rasters...'
        #Write GLCM rasters to file
        make_glcm_raster(ent,var,homo,v1,v2,v3)
        
        
        
        
        
        
        
        