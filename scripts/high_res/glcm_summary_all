# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 12:12:42 2016

@author: dan
"""
from rasterstats import zonal_stats

if __name__ == '__main__':
    
    #input rasters
    ent_dict = {'R01346':r"C:\workspace\GLCM\output\new_glcm_rasters\2014_04\3\R01346_R01347_3_entropy_resampled.tif",
                'R01765':r"C:\workspace\GLCM\output\new_glcm_rasters\2014_09_2\3\R01765_3_entropy_resampled.tif",
                'R01767':r"C:\workspace\GLCM\output\new_glcm_rasters\2014_09\3\R01767_3_entropy_resampled.tif"}
                
    var_dict = {'R01346':r"C:\workspace\GLCM\output\new_glcm_rasters\2014_04\3\R01346_R01347_3_var_resampled.tif",
                'R01765':r"C:\workspace\GLCM\output\new_glcm_rasters\2014_09_2\3\R01765_3_var_resampled.tif",
                'R01767':r"C:\workspace\GLCM\output\new_glcm_rasters\2014_09\3\R01767_3_var_resampled.tif"}       
    
    homo_dict = {'R01346':r"C:\workspace\GLCM\output\new_glcm_rasters\2014_04\3\R01346_R01347_3_homo_resampled.tif",
                'R01765':r"C:\workspace\GLCM\output\new_glcm_rasters\2014_09_2\3\R01765_3_homo_resampled.tif",
                'R01767':r"C:\workspace\GLCM\output\new_glcm_rasters\2014_09\3\R01767_3_homo_resampled.tif"}  
    
    shp_dict = {'R01346':r"C:\workspace\Merged_SS\window_analysis\shapefiles\tex_seg_800_3class.shp",
                'R01765':r"C:\workspace\Merged_SS\window_analysis\shapefiles\R01765.shp",
                'R01767':r"C:\workspace\Merged_SS\window_analysis\shapefiles\tex_seg_2014_09_67_3class.shp"}  
    
    for (k,v), (k1,v1), (k2,v2), (k3,v3) in zip(ent_dict.items(),var_dict.items(), homo_dict.items(),shp_dict,items()):
        ent_raster = v
        var_raster = v1
        homo_raster= v2
        in_shp = v3
        
        #Get mini rasters
        ent_stats = zonal_stats(in_shp, ent_raster, stats=['count','mean'], raster_out=True)
        var_stats = zonal_stats(in_shp, var_raster, stats=['count','mean'], raster_out=True)
        homo_stats = zonal_stats(in_shp, homo_raster, stats=['count','mean'], raster_out=True)