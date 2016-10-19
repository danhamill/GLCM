# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 09:27:12 2016

@author: dan
"""


import numpy as np
import gdal
import ogr
import osr
from rasterstats import zonal_stats
import pandas as pd
import matplotlib.pyplot as plt
import os

def read_raster(in_raster):
    '''
    Function to open input raster
    
    Returns
    
    data = data contained in the raster
    gt = geotransform of input raster
    xx =  numpy mesh grid of easting corrdinates
    yy = numpy mesh grid of northing cordinates
    '''
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
    
def CreateRaster(xx,yy,std,gt,outFile):  
    '''
    Exports data to GTiff Raster
    '''
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(26949)
    std = np.squeeze(std)

    driver = gdal.GetDriverByName('GTiff')
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

def cat_data(data):
    data[np.where((0 < data) & (data < 2.75))] = 1
    data[np.where((2.75 < data) & (data < 4)) ] = 2
    data[np.where(4 < data) ] = 3
    data[np.isnan(data)] = -99
    return data


win_sizes = [8,12,20,40,80]
for win_size in win_sizes:
    #Stuff to change
    win = win_size
    meter = str(win/4)
    in_raster = r"C:\workspace\GLCM\output\2014_09" + os.sep + meter +os.sep+"R01767_" + meter + ".tif"
    
    oName = r"C:\workspace\GLCM\output\2014_09" + os.sep + meter +os.sep+"R01767_" + meter + "_classified.tif"
    
    if  not os.path.exists(oName):       
        #Read in unlassified raster
        data, xx, yy, gt = read_raster(in_raster)
    
        #Bin data in to catagories
        data = cat_data(data)
    
        #Create classified raster
        CreateRaster(xx,yy,data,gt,oName)  
        print 'Created catagorized raster!!!'
    else:
        pass
       
    #Read in catagorical raster for zonal stats
    data, xx, yy, gt = read_raster(oName)

    #Segmetion shpaefile
    in_shp = r"C:\workspace\Merged_SS\window_analysis\shapefiles\tex_seg_2014_09_67_3class.shp"

    #Calculate Zonal Statistics
    stats = zonal_stats(in_shp, oName, categorical=True, nodata=-99)
    
    #Lets get area and substate codes
    ds = ogr.Open(in_shp)
    lyr = ds.GetLayer(0)
    a=[]
    b = []
    for row in lyr:
        geom = row.GetGeometryRef()
        a.append(row.substrate)
        b.append(geom.GetArea())
    lyr.ResetReading()
    del ds
    
    
    t = pd.DataFrame(stats)
    t['substrate'] = a

    #Clean up workspace 
    del  a, b,in_shp, data, xx, yy, gt, oName, in_raster

    #April 2014
    in_raster = r"C:\workspace\GLCM\output\2014_04" + os.sep + meter +os.sep+"R01346_R01347_" + meter + ".tif"
    
    oName = r"C:\workspace\GLCM\output\2014_04" + os.sep + meter +os.sep+"R01346_R01347_" + meter + "_classified.tif"
    if  not os.path.exists(oName):       
        #Read in unlassified raster
        data, xx, yy, gt = read_raster(in_raster)
        
        #Bin data in to catagories
        data = cat_data(data)
    
        #Create classified raster
        CreateRaster(xx,yy,data,gt,oName)  
        print 'Created catagorized raster!!!'
    else:
        print 'Raster Already exists.'
        pass
       
    data, xx, yy, gt = read_raster(oName)

    
    in_shp = r"C:\workspace\Merged_SS\window_analysis\shapefiles\tex_seg_800_3class.shp"
    
    stats = zonal_stats(in_shp, oName, categorical=True, nodata=-99)
    
    ds = ogr.Open(in_shp)
    lyr = ds.GetLayer(0)
    a=[]
    b = []
    for row in lyr:
        geom = row.GetGeometryRef()
        a.append(row.substrate)
        b.append(geom.GetArea())
    lyr.ResetReading()
    del ds
    
    
    df = pd.DataFrame(stats)
    df['substrate'] = a
    
    fig, (ax1,ax2,ax3) = plt.subplots(nrows=3,figsize=(4,6))
    pd.pivot_table(t, index=['substrate'],values=[1.0,2.0,3.0],aggfunc = np.sum).plot.bar(ax=ax1,title='2014_09',rot=0, xlim=(0,150000))
    pd.pivot_table(df, index=['substrate'],values=[1.0,2.0,3.0],aggfunc = np.sum).plot.bar(ax=ax2,title='2014_04',rot=0, xlim=(0,500000))
    
    merge = pd.concat([t,df])
    
    pd.pivot_table(df, index=['substrate'],values=[1.0,2.0,3.0],aggfunc = np.sum).plot.bar(ax=ax3,title='Combined',rot=0, xlim=(0,500000))
    
    plt.tight_layout()
    ax1.legend(loc='9', ncol=3, columnspacing=1, fontsize=8)
    ax2.legend(loc='9', ncol=3, columnspacing=1, fontsize=8)
    ax3.legend(loc='9', ncol=3, columnspacing=1, fontsize=8)
    root = r"C:\workspace\GLCM\output\std_plots"
    
    fig_name = root + os.sep + 'STD_' + meter + '_catagorized.png'
    plt.savefig(fig_name, dpi = 600)
    
    #del  a, b,in_shp, data, xx, yy, gt, oName, in_raster, df ,t