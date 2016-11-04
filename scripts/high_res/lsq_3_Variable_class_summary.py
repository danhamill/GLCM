# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 11:31:38 2016

@author: dan
"""

from rasterstats import zonal_stats
import ogr
import pandas as pd
import numpy as np


def get_subs(shp):
    ds = ogr.Open(shp)
    lyr = ds.GetLayer(0)
    a=[]
    for row in lyr:
        a.append(row.substrate)
    lyr.ResetReading()
    del ds
    return a

def df_to_markdown(df, float_format='%.2g'):
    """
    Export a pandas.DataFrame to markdown-formatted text.
    DataFrame should not contain any `|` characters.
    """
    from os import linesep
    return linesep.join([
        '|'.join(df.columns),
        '|'.join(4 * '-' for i in df.columns),
        df.to_csv(sep='|', index=False, header=False, float_format=float_format)
    ]).replace('|', ' | ')
    
rasters = [r"C:\workspace\GLCM\output\least_sqares_classification\R01346_R01347_percentile50_Sed_Class_3_variable.tif", 
r"C:\workspace\GLCM\output\least_sqares_classification\R01765_percentile50_Sed_Class_3_variable.tif", 
r"C:\workspace\GLCM\output\least_sqares_classification\R01767_percentile50_Sed_Class_3_variable.tif"]

shps = [r"C:\workspace\Merged_SS\window_analysis\shapefiles\tex_seg_800_3class.shp", r"C:\workspace\Merged_SS\window_analysis\shapefiles\R01765.shp", r"C:\workspace\Merged_SS\window_analysis\shapefiles\tex_seg_2014_09_67_3class.shp"]

#n=0
for n in xrange(len(rasters)):
    raster = rasters[n]
    shp = shps[n]
    subs = get_subs(shp)
    stats = zonal_stats(shp, raster, categorical=True, nodata=-99)
    if raster == rasters[0]:
        merge_subs = subs
        merge_df = pd.DataFrame(stats)
    else:
        merge_subs.extend(subs)
        merge_df = pd.concat([merge_df,pd.DataFrame(stats)])
    del stats, shp,raster,subs
    

merge_df['substrate'] = merge_subs
#merge_df['Sand'] = 0
#merge_df = merge_df.fillna(0)
merge_df.rename(columns = {0.0:'null',1.0:'Sand',2.0:'Gravel',3.0:'Boulders'},inplace=True)
merge_df = merge_df[['null','Sand','Gravel','Boulders','substrate']]
pvt = pd.pivot_table(merge_df, index=['substrate'],values=['null','Sand','Gravel','Boulders'],aggfunc=np.nansum)

#Percentage classification table
df_to_markdown(pvt.div(pvt.sum(axis=1), axis=0))




'''
homo_stats = zonal_stats(in_shp, homo_raster,stats=['min','mean','max','median','std','count','percentile_25','percentile_50','percentile_75'])
ent_stats = zonal_stats(in_shp, ent_raster,stats=['min','mean','max','median','std','count','percentile_25','percentile_50','percentile_75'])
var_stats = zonal_stats(in_shp, var_raster,stats=['min','mean','max','median','std','count','percentile_25','percentile_50','percentile_75'])

in_shp = r"C:\workspace\Merged_SS\window_analysis\shapefiles\tex_seg_2014_09_67_3class.shp"
ent_raster = r"C:\workspace\GLCM\output\glcm_rasters\2014_09\3\R01767_3_entropy_resampled.tif"
var_raster = r"C:\workspace\GLCM\output\glcm_rasters\2014_09\3\R01767_3_var_resampled.tif"
homo_raster = r"C:\workspace\GLCM\output\glcm_rasters\2014_09\3\R01767_3_homo_resampled.tif"
'''

