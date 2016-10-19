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
    in_raster = r"C:\workspace\Merged_SS\raster\2014_09\ss_2014_09_R01767_raster.tif"
    win = win_size
    meter = str(win/4)
    
    
    oName = r"C:\workspace\GLCM\output\2014_09" + os.sep + meter +os.sep+"R01767_" + meter + "_CV_classified.tif"

    
    #Read in unlassified raster
    data, xx, yy, gt = read_raster(in_raster)

    #Bin data in to catagories
    data = cat_data(data)

    

    CreateRaster(xx,yy,data,gt,oName)

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





t = df

del df, a, b,in_shp, data, xx, yy, gt, oName, in_raster

in_raster = r"C:\workspace\Merged_SS\raster\2014_09\ss_2014_09_R01767_raster.tif"
    

data, xx, yy, gt = read_raster(in_raster)

data = cat_data(data)

outFile = r"C:\workspace\GLCM\output\2014_04" + os.sep + meter +os.sep+"R01346_R01347_" + meter + "_CV.tif"
#CreateRaster(xx,yy,data,gt,oName)

data, xx, yy, gt = read_raster(oName)
data[np.isnan(data)] = -99

in_shp = r"C:\workspace\Merged_SS\window_analysis\shapefiles\tex_seg_2014_09_67_3class.shp"


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

fig, (ax1,ax2,ax3) = plt.subplots(nrows=3,figsize=(6,6))
pd.pivot_table(t, index=['substrate'],values=[1.0,2.0,3.0],aggfunc = np.sum).plot.bar(ax=ax1,title='2014_04')
pd.pivot_table(df, index=['substrate'],values=[1.0,2.0,3.0],aggfunc = np.sum).plot.bar(ax=ax2,title='2014_04')

merge = pd.concat([t,df])

pd.pivot_table(df, index=['substrate'],values=[1.0,2.0,3.0],aggfunc = np.sum).plot.bar(ax=ax3,title='Combined')

plt.tight_layout()
plt.show()