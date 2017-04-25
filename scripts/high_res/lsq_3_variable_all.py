# -*- coding: utf-8 -*-

import os
import osr
from osgeo import gdal
from osgeo import ogr
import numpy as np
import pandas as pd
from scikits.bootstrap import bootstrap as boot
from rasterstats import zonal_stats
import pytablewriter
import itertools
from sklearn.metrics import cohen_kappa_score, classification_report

# =========================================================
def get_subs(shp):
    ds = ogr.Open(shp)
    lyr = ds.GetLayer(0)
    a=[]
    for row in lyr:
        a.append(row.substrate)
    lyr.ResetReading()
    del ds
    return a
  # =========================================================  
def read_raster(raster):
    ds = gdal.Open(raster)
    data = ds.GetRasterBand(1).ReadAsArray()
    gt = ds.GetGeoTransform()
    return data, gt


# =========================================================
def lsqnonneg(C, d, x0=None, tol=None, itmax_factor=3):
    '''Linear least squares with nonnegativity constraints

    (x, resnorm, residual) = lsqnonneg(C,d) returns the vector x that minimizes norm(d-C*x)
    subject to x >= 0, C and d must be real
    '''

    eps = 2.22e-16    # from matlab
    def norm1(x):
        return abs(x).sum().max()

    def msize(x, dim):
        s = x.shape
        if dim >= len(s):
            return 1
        else:
            return s[dim]

    if tol is None:
        tol = 10*eps*norm1(C)*(max(C.shape)+1)

    C = np.asarray(C)

    (m,n) = C.shape
    P = np.zeros(n)
    Z = np.arange(1, n+1)

    if x0 is None:
        x=P
    else:
        if any(x0 < 0):
            x=P
        else:
            x=x0

    ZZ=Z

    resid = d - np.dot(C, x)
    w = np.dot(C.T, resid)

    outeriter=0
    it=0
    itmax=itmax_factor*n
    exitflag=1

    # outer loop to put variables into set to hold positive coefficients
    while np.any(Z) and np.any(w[ZZ-1] > tol):
        outeriter += 1

        t = w[ZZ-1].argmax()
        t = ZZ[t]

        P[t-1]=t
        Z[t-1]=0

        PP = np.where(P <> 0)[0]+1
        ZZ = np.where(Z <> 0)[0]+1

        CP = np.zeros(C.shape)

        CP[:, PP-1] = C[:, PP-1]
        CP[:, ZZ-1] = np.zeros((m, msize(ZZ, 1)))

        z=np.dot(np.linalg.pinv(CP), d)

        z[ZZ-1] = np.zeros((msize(ZZ,1), msize(ZZ,0)))

        # inner loop to remove elements from the positve set which no longer belong
        while np.any(z[PP-1] <= tol):
            if it > itmax:
                max_error = z[PP-1].max()
                raise Exception('Exiting: Iteration count (=%d) exceeded\n Try raising the tolerance tol. (max_error=%d)' % (it, max_error))

            it += 1

            QQ = np.where((z <= tol) & (P <> 0))[0]
            alpha = min(x[QQ]/(x[QQ] - z[QQ]))
            x = x + alpha*(z-x)

            ij = np.where((abs(x) < tol) & (P <> 0))[0]+1
            Z[ij-1] = ij
            P[ij-1] = np.zeros(max(ij.shape))
            PP = np.where(P <> 0)[0]+1
            ZZ = np.where(Z <> 0)[0]+1

            CP[:, PP-1] = C[:, PP-1]
            CP[:, ZZ-1] = np.zeros((m, msize(ZZ, 1)))

            z=np.dot(np.linalg.pinv(CP), d)
            z[ZZ-1] = np.zeros((msize(ZZ,1), msize(ZZ,0)))

        x = z
        resid = d - np.dot(C, x)
        w = np.dot(C.T, resid)

    return (x, sum(resid * resid), resid)


# =========================================================
def get_class(calib,vec,w):
   '''
   return the percent variance associated with sand, gravel and rock, and the residual norm
   '''
#   vec = (0,vec1[ind[k]],vec2[ind[k]],vec3[ind[k]])
#   calib = calib
   X = lsqnonneg(calib,vec, x0=np.zeros(np.shape(calib.T)[0]))
   dist = (X[0]*w)/np.sum(X[0]*w)
   prc_sand = dist[0]
   prc_gravel = dist[1]
   prc_rock = dist[2]
   ss_resid = X[1] 
   return prc_sand, prc_gravel, prc_rock, ss_resid
 
# =========================================================
def make_class(row):
    if row['sand_conf'] >= 0.25 and row['gravel_conf']<0.25 and row['rock_conf']<0.25:
        return 1
    if row['sand_conf'] < 0.25 and row['gravel_conf']>=0.25 and row['rock_conf']<0.25:
        return 2
    if row['sand_conf'] < 0.25 and row['gravel_conf']<0.25 and row['rock_conf']>=0.25:
        return 3
    if row['sand_conf'] < 0.25 and row['gravel_conf']<0.25 and row['rock_conf']<0.25:
        return 0
    if np.isnan(row['sand_conf']) and np.isnan(row['gravel_conf']) and np.isnan(row['rock_conf']):
        return np.nan

# =========================================================
def CreateRaster(sed_class,gt,outFile):  
    '''
    Exports data to GTiff Raster
    '''
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(26949)
    sed_class = np.squeeze(sed_class)
    sed_class[np.isnan(sed_class)] = -99
    driver = gdal.GetDriverByName('GTiff')
    rows,cols = np.shape(sed_class)
    ds = driver.Create( outFile, cols, rows, 1, gdal.GDT_Float32)      
    if proj is not None:  
        ds.SetProjection(proj.ExportToWkt()) 
    ds.SetGeoTransform(gt)
    ss_band = ds.GetRasterBand(1)
    ss_band.WriteArray(sed_class)
    ss_band.SetNoDataValue(-99)
    ss_band.FlushCache()
    ss_band.ComputeStatistics(False)
    del ds
      
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
    
    #Input zonal statistics            
    homo = r"C:\workspace\GLCM\new_output\homo_3_zonal_stats_merged.csv"
    var = r"C:\workspace\GLCM\new_output\var_3_zonal_stats_merged.csv"
    ent = r"C:\workspace\GLCM\new_output\entropy_3_zonal_stats_merged.csv"
        
    df1 = pd.read_csv(homo,sep=',')
    df2 = pd.read_csv(ent,sep=',')
    df3 = pd.read_csv(var,sep=',')
    del homo,var,ent
    df1.rename(columns={'max':'homo_max', 'mean':'homo_mean', 'median':'homo_median','min':'homo_min','percentile_25':'homo_25','percentile_50':'homo_50', 'percentile_75':'homo_75','std':'homo_std'},inplace=True)   
    df2.rename(columns={'max':'entropy_max', 'mean':'entropy_mean', 'median':'entropy_median','min':'entropy_min','percentile_25':'entropy_25','percentile_50':'entropy_50', 'percentile_75':'entropy_75','std':'entropy_std'},inplace=True)   
    df3.rename(columns={'max':'var_max', 'mean':'var_mean', 'median':'var_median','min':'var_min','percentile_25':'var_25','percentile_50':'var_50', 'percentile_75':'var_75','std':'var_std'},inplace=True)   
    
    
    merge =df1.merge(df2,left_index=True, right_index=True, how='left')
    merge = merge.merge(df3,left_index=True, right_index=True, how='left' )
    merge = merge[['homo_median','entropy_median','var_median','substrate']].dropna()
    merge.rename(columns={'substrate_x':'substrate'},inplace=True)
    del df1,df2,df3
    grouped = merge.groupby('substrate')
    sand = grouped.get_group('sand')
    gravel = grouped.get_group('gravel')
    boulders = grouped.get_group('boulders')
    del merge
    
    calib_df = pd.DataFrame(columns=['ent','homo','var'], index=['sand','gravel','boulders'])

    calib_df.loc['sand'] = pd.Series({'homo':1- np.average(boot.ci(sand['homo_median'],np.median,alpha=0.05)) ,
                                    'ent':np.average(boot.ci(sand['entropy_median'],np.median,alpha=0.05)) ,
                                    'var': np.average(boot.ci(sand['var_median'],np.median,alpha=0.05))})
    calib_df.loc['gravel'] = pd.Series({'homo':1- np.average(boot.ci(gravel['homo_median'],np.median,alpha=0.05)) ,
                                    'ent':np.average(boot.ci(gravel['entropy_median'],np.median,alpha=0.05)) ,
                                    'var': np.average(boot.ci(gravel['var_median'],np.median,alpha=0.05))})
    calib_df.loc['boulders'] = pd.Series({'homo':1- np.average(boot.ci(boulders['homo_median'],np.median,alpha=0.05)) ,
                                    'ent':np.average(boot.ci(boulders['entropy_median'],np.median,alpha=0.05)) ,
                                    'var': np.average(boot.ci(boulders['var_median'],np.median,alpha=0.05))})
    
    del sand, gravel, boulders         
                
                
    



  
        
        
    for (k,v), (k1,v1), (k2,v2) in zip(ent_dict.items(),var_dict.items(), homo_dict.items()):
        ent_raster = v
        var_raster = v1
        homo_raster= v2

        #Get the data
        ent_data, gt = read_raster(ent_raster)
        var_data = read_raster(var_raster)[0]
        homo_data = read_raster(homo_raster)[0]
        df = pd.DataFrame({'ent':ent_data.flatten(),'var':var_data.flatten(),'homo':homo_data.flatten()})
    
        outFile = r"C:\workspace\GLCM\output\least_sqares_classification" + os.sep + k +"_median_Sed_Class_3_variable.tif"
        #======================================================
        ## inputs
        w = [0.1,0.7,0.2] #weightings - leave at 1 unless you have any preference for 1 input variable over another. 
        
        calib = np.asarray(calib_df.values,dtype='float')
        vec1 = ent_data.flatten()#flattened array of homogeneity values from a given sample (sidescan)
        vec2 = 1-homo_data.flatten()#flattened array of entropy values
        vec3 = var_data.flatten()#flattened array of GLCM variance values
        ##vec4 = #??
        
        vec1[np.isnan(vec1)] = 0; vec2[np.isnan(vec2)] = 0; vec3[np.isnan(vec3)] = 0
        vec1[np.isinf(vec1)] = 0; vec2[np.isinf(vec2)] = 0; vec3[np.isinf(vec3)] = 0
        ind = np.nonzero(vec1)[0]
        # =============== 
        # classify!
        # pre-allocate arrays
        prc_sand = np.zeros(np.shape(vec1))*np.nan
        prc_gravel = np.zeros(np.shape(vec1))*np.nan
        prc_rock = np.zeros(np.shape(vec1))*np.nan
        ss_resid = np.zeros(np.shape(vec1))*np.nan # residual norm
        
        
        # classify 
        for k in xrange(len(ind)):
              prc_sand[ind[k]], prc_gravel[ind[k]], prc_rock[ind[k]], ss_resid[ind[k]] = get_class(calib.T,(vec1[ind[k]],vec2[ind[k]],vec3[ind[k]]),w)
        
        # now reshape the arrays
        
        # =============== reshape
        Ny, Nx = np.shape(ent_data)
        prc_sand = np.reshape(prc_sand,(Ny, Nx))
        prc_gravel = np.reshape(prc_gravel,(Ny, Nx))
        prc_rock = np.reshape(prc_rock,(Ny, Nx))
        ss_resid = np.reshape(ss_resid,(Ny, Nx))
        
        # =============== define confidence metric
        sand_conf = prc_sand*(1-prc_rock)*(1-prc_gravel)
        rock_conf = prc_rock*(1-prc_sand)*(1-prc_gravel)
        gravel_conf = prc_gravel*(1-prc_sand)*(1-prc_rock)
          
        sed_df = pd.DataFrame({'prc_sand':prc_sand.flatten(),'sand_conf':sand_conf.flatten(),'prc_gravel':prc_gravel.flatten(),
        'gravel_conf':gravel_conf.flatten(),'prc_rock':prc_rock.flatten(),'rock_conf':rock_conf.flatten()})
        
        sed_df = sed_df[['prc_sand','sand_conf','prc_gravel','gravel_conf','prc_rock','rock_conf']]
        
        sed_df['sedclass'] = sed_df.apply(lambda row: make_class(row),axis=1)
        
        sed_class = np.reshape(np.asarray(sed_df['sedclass'],dtype='float'),np.shape(sand_conf))
        
        CreateRaster(sed_class,gt,outFile)

        del k, k1, k2, v1, v, v2, sed_class, sed_df, sand_conf, rock_conf, gravel_conf,prc_sand,prc_gravel,prc_rock,ss_resid,vec1,vec2,vec3,outFile,homo_data,var_data,ent_data,ind,gt,Nx,Ny,ent_raster,homo_raster,var_raster

    #classification results       
    rasters = [r"C:\workspace\GLCM\output\least_sqares_classification\R01346_median_Sed_Class_3_variable.tif", 
               r"C:\workspace\GLCM\output\least_sqares_classification\R01765_median_Sed_Class_3_variable.tif", 
               r"C:\workspace\GLCM\output\least_sqares_classification\R01767_median_Sed_Class_3_variable.tif"]
    
    shps = [r"C:\workspace\Merged_SS\window_analysis\shapefiles\tex_seg_800_3class.shp", 
            r"C:\workspace\Merged_SS\window_analysis\shapefiles\R01765.shp", 
            r"C:\workspace\Merged_SS\window_analysis\shapefiles\tex_seg_2014_09_67_3class.shp"]
    
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
        
    del n
    
    merge_df['substrate'] = merge_subs
    
    merge_df.rename(columns = {0.0:'null',1.0:'Sand',2.0:'Gravel',3.0:'Boulders'},inplace=True)
    merge_df = merge_df[['null','Sand','Gravel','Boulders','substrate']]
    
    pvt = pd.pivot_table(merge_df, index=['substrate'],values=['null','Sand','Gravel','Boulders'],aggfunc=np.nansum)
    del merge_df
    #Percentage classification table
    class_df = pvt.div(pvt.sum(axis=1), axis=0)

    writer = pytablewriter.MarkdownTableWriter()
    writer.table_name = "Non-negative Least Squares Classification Results"
    writer.header_list = list(class_df.columns.values)
    writer.value_matrix = class_df.values.tolist()
    writer.write_table()   
        
    
    df = pd.pivot_table(merge_df,values=['Sand','Gravel','Boulders'],index=['substrate'],aggfunc=np.sum)

    df = pd.read_csv(r"C:\workspace\GLCM\new_output\LSQ_Results\pred_true_lsq.csv",sep=',',names=['pred','true'],header=None)
    print(classification_report(df['pred'].ravel(),df['true'].ravel()))
    print (cohen_kappa_score(df['pred'].ravel(),df['true'].ravel()))