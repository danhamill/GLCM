# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 12:12:42 2016

@author: dan
"""
from rasterstats import zonal_stats
import pandas as pd
from osgeo import ogr
import os



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
    fnames = []
    for (k,v), (k1,v1), (k2,v2), (k3,v3) in zip(ent_dict.items(),var_dict.items(), homo_dict.items(),shp_dict.items()):
        
        ent_raster = v
        var_raster = v1
        homo_raster= v2
        in_shp = v3
        
        #Get mini rasters
        ent_stats = zonal_stats(in_shp, ent_raster, stats=['count','mean'], raster_out=True)
        var_stats = zonal_stats(in_shp, var_raster, stats=['count','mean'], raster_out=True)
        homo_stats = zonal_stats(in_shp, homo_raster, stats=['count','mean'], raster_out=True)
        
        #Aggregrate based on 
        s_ent,  g_ent, r_ent, a = agg_distributions(ent_stats, in_shp,'Entropy')
        s_var,  g_var, r_var = agg_distributions(var_stats, in_shp,'Variance')[0:3]
        s_homo,  g_homo, r_homo = agg_distributions(homo_stats, in_shp,'Homogeneity')[0:3]
        del ent_stats, var_stats, homo_stats
        
        s_df = pd.concat([s_ent,pd.concat([s_var,s_homo],axis=1)],axis=1)
        g_df = pd.concat([g_ent,pd.concat([g_var,g_homo],axis=1)],axis=1)
        r_df = pd.concat([r_ent,pd.concat([r_var,r_homo],axis=1)],axis=1)
        del s_ent, g_ent, r_ent, s_var, g_var, r_var, s_homo, g_homo, r_homo
        
        s_df['sedclass'] = 1
        g_df['sedclass'] = 2
        r_df['sedclass'] = 3

        agg_dist = pd.concat([s_df,pd.concat([g_df,r_df])])
        oName = r"C:\workspace\GLCM\new_output" + os.sep + k + "_aggregraded_distributions.csv"
        fnames.append(oName)
        agg_dist.to_csv(oName,sep=',',index=False)
        
    del (k,v), (k1,v1), (k2,v2), (k3,v3), oName, agg_dist, ent_raster,var_raster,homo_raster,in_shp,s_df,g_df,r_df

    df1 = pd.read_csv(fnames[0],sep=',')  
    df2 = pd.read_csv(fnames[1],sep=',')  
    df3 = pd.read_csv(fnames[2],sep=',')
    
    merge_dist = pd.concat([df1,pd.concat([df2,df3])])
    del df1,df2,df3
    oName = r"C:\workspace\GLCM\new_output" + os.sep + "merged_aggregraded_distributions.csv"
    merge_dist.to_csv(oName,sep=',',index=False)
        
        
        