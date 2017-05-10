# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 09:55:15 2017

@author: dan
"""

from osgeo import gdal
import numpy as np
import pandas as pd
import os
from skimage.segmentation import slic, mark_boundaries
import matplotlib.pyplot as plt
import pytablewriter


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
               'R01349':r"C:\workspace\Merged_SS\raster\2014_04\ss_2014_04_R01349_clip_raster.tif",
               "R01350": r"C:\workspace\Merged_SS\raster\2014_04\ss_2014_04_R01350_clip_raster.tif",
                'R01765':r"C:\workspace\Merged_SS\raster\2014_09\ss_2014_09_R01765_raster.tif",
                'R01767':r"C:\workspace\Merged_SS\raster\2014_09\ss_2014_09_R01767_raster.tif"}
    
    n_seg_dict = {'R01346':900,
               'R01349':700,
               "R01350": 300,
                'R01765':125,
                'R01767':900}
                                 

    for (k,v), (k1,v1), in zip(ss_dict.items(),n_seg_dict.items()):
        
        
        ss_raster = v
        num = v1

        #Find segments for GLCM calculations   
        im, xx, yy, gt = read_raster(ss_raster)
        im[np.isnan(im)] = 0
        im = rescale(im,0,1)
        
        #initialize segments for iteration
        print 'Now calculating n segments for slic segments for %s...' %(k,)
        segments_slic = slic(im, n_segments=int(num), compactness=0.1,slic_zero=False) 
        
        oName = r"C:\workspace\GLCM\slic_output\slic_segments_plots\compactness_0.1" + os.sep + k + "_" + str(num) + "_slic_segments.png"
        
        if os.path.exists(oName):
            pass
        else:
            fig,ax = plt.subplots()
            ax.imshow(mark_boundaries(im, segments_slic,color=[1,0,0])) 
            title = k + ' n_segments = %s' %(str(num),)
            ax.set_title(title)
            plt.tight_layout()
            plt.savefig(oName,dpi=600)
            plt.close()
            
        thing = segments_slic.copy()
        thing = thing.astype('float')
        thing[np.isnan(read_raster(ss_raster)[0])] = np.nan
        
        thing = thing[~np.isnan(thing)]
        
        unique, counts = np.unique(thing, return_counts=True)
        
        areas = np.zeros(np.shape(unique),dtype='float64')
        
        areas = counts * 0.25**2
        
        try:
            len(df)
        except:
            df = pd.DataFrame(index=['R01346','R01349','R01350','R01765','R01767'],columns=['Length','Area'])
            df['Length'] = [646,422,290,183,544]
            
            
        df.loc[k,'Area'] = np.average(areas, axis=0)
    
    df.loc[:,'N_Segments'] = [900,700,300,125,900]
    
    writer = pytablewriter.MarkdownTableWriter()
    writer.table_name = "SLIC n_segments analysis"
    writer.header_list = list(df.reset_index().columns.values)
    writer.value_matrix = df.reset_index().values.tolist()
    writer.write_table()   

    df = df.sort_values(by=['N_Segments'])         

    fig,ax = plt.subplots()
    df.plot(ax = ax,kind='line', x='N_Segments', y='Length',marker='x',xlim=(100,1000),ylim=(100,700),legend=False   )                                                                     
    ax.set_ylabel('Scan Length (m)')
    plt.tight_layout()