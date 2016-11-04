# -*- coding: utf-8 -*-
"""
Created on Fri Nov 04 10:53:10 2016

@author: dan
"""

import osr
import gdal
import numpy as np
import pandas as pd

def centeroidnp(df,df1,df2,df3,query1,metric):
#    length = df.query(query1)[metric].dropna().values.size
#    sum_x = np.nansum(df.query(query1)[metric].values)
#    sum_y = np.nansum(df1.query(query1)[metric].values)
#    sum_z = np.nansum(df2.query(query1)[metric].values)
#    return sum_x/length, sum_y/length, sum_z/length
    
    x = np.nanmedian(df.query(query1)[metric])
    y = np.nanmedian(df1.query(query1)[metric])
    z= np.nanmedian(df2.query(query1)[metric])
    m = np.nanmedian(df3.query(query1)[metric])
    return x,y,z, m 
    
def error_bars(df,df1,df2,df3,query1,metric):
    y_err = np.nanstd(df.query(query1)[metric].values)
    x_err = np.nanstd(df1.query(query1)[metric].values)
    z_err = np.nanstd(df2.query(query1)[metric].values)
    m_err = np.nanstd(df3.query(query1)[metric].values)
    return x_err, y_err, z_err, m_err

def read_raster(raster):
    ds = gdal.Open(raster)
    data = ds.GetRasterBand(1).ReadAsArray()
    gt = ds.GetGeoTransform()
    return data, gt

def get_center(cont_df,var_df,ent_df,homo_df,metric):
    s_query = 'substrate == "sand"'
    g_query = 'substrate == "gravel"'
    b_query = 'substrate == "boulders"'
    s_centroid = centeroidnp(cont_df,var_df,ent_df,homo_df,s_query,metric)    
    g_centroid = centeroidnp(cont_df,var_df,ent_df,homo_df,g_query,metric)    
    b_centroid = centeroidnp(cont_df,var_df,ent_df,homo_df,b_query,metric)    
    s_err = error_bars(cont_df,var_df,ent_df,homo_df,s_query,metric)    
    g_err = error_bars(cont_df,var_df,ent_df,homo_df,g_query,metric)
    b_err = error_bars(cont_df,var_df,ent_df,homo_df,b_query,metric)
    cent_df = pd.DataFrame(columns=['x_cent','y_cent','z_cent','m_cent','y_err','x_err','z_err','m_err'], index=['sand','gravel','boulders'])
    #cent_df.loc['null'] = pd.Series({'x_cent':0. ,'y_cent':0. ,'z_cent':0.,'m_cent':0.,'y_err':0. ,'x_err':0.,'z_err':0.,'m_err':0.})
    cent_df.loc['sand'] = pd.Series({'x_cent':s_centroid[0] ,'y_cent':s_centroid[1] ,'z_cent':s_centroid[2],'m_cent':1-s_centroid[3],'y_err':s_err[1] ,'x_err':s_err[0],'z_err':s_err[2], 'm_err':s_err[3]})
    cent_df.loc['gravel'] = pd.Series({'x_cent':g_centroid[0] ,'y_cent': g_centroid[1],'z_cent':g_centroid[2],'m_cent':1-g_centroid[3],'y_err':g_err[1] ,'x_err':g_err[0],'z_err':g_err[2],'m_err':b_err[3]})
    cent_df.loc['boulders'] = pd.Series({'x_cent':b_centroid[0] ,'y_cent':b_centroid[1] ,'z_cent':b_centroid[2],'m_cent':1-b_centroid[3],'y_err': b_err[1],'x_err':b_err[0],'z_err':b_err[2],'m_err':g_err[3]})
    cent_df = cent_df[['z_cent','x_cent','y_cent','m_cent']]#
    return cent_df

#   d= (0,vec1[ind[k]],vec2[ind[k]],vec3[ind[k]],vec4[ind[k]])
#   C = calib.T
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

   X = lsqnonneg(calib,vec, x0=np.zeros(np.shape(calib.T)[0]))
   dist = (X[0]*w)/np.sum(X[0]*w)
   prc_sand = dist[0]
   prc_gravel = dist[1]
   prc_rock = dist[2]
   ss_resid = X[1] 
   return prc_sand, prc_gravel, prc_rock, ss_resid
   
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
    
    ent_raster = r"C:\workspace\GLCM\output\new_glcm_rasters\2014_04\3\R01346_R01347_3_entropy_resampled.tif"
    var_raster = r"C:\workspace\GLCM\output\new_glcm_rasters\2014_04\3\R01346_R01347_3_var_resampled.tif"
    cont_raster = r"C:\workspace\GLCM\output\new_glcm_rasters\2014_04\3\R01346_R01347_3_contrast_resampled.tif"
    homo_raster = r"C:\workspace\GLCM\output\new_glcm_rasters\2014_04\3\R01346_R01347_3_homo_resampled.tif"
       
    cont = r"C:\workspace\GLCM\new_output\contrast_3_zonal_stats_merged.csv"
    var = r"C:\workspace\GLCM\new_output\var_3_zonal_stats_merged.csv"
    ent = r"C:\workspace\GLCM\new_output\entropy_3_zonal_stats_merged.csv"
    homo = r"C:\workspace\GLCM\new_output\homo_3_zonal_stats_merged.csv"   
    
    cont_df = pd.read_csv(cont,sep=',',usecols=['percentile_25','percentile_50','percentile_75','substrate'])
    var_df = pd.read_csv(var, sep=',',usecols=['percentile_25','percentile_50','percentile_75','substrate'])
    ent_df = pd.read_csv(ent, sep=',',usecols=['percentile_25','percentile_50','percentile_75','substrate'])
    homo_df = pd.read_csv(homo, sep=',',usecols=['percentile_25','percentile_50','percentile_75','substrate'])
    

    #Get the data
    ent_data, gt = read_raster(ent_raster)
    var_data = read_raster(var_raster)[0]
    cont_data = read_raster(cont_raster)[0]
    homo_data = read_raster(homo_raster)[0]
    
#    data_merge = pd.DataFrame({'ent':ent_data.flatten(),'var':var_data.flatten(),'homo':homo_data.flatten(),'cont_data':cont_data.flatten()})
#    s_query = 'substrate == "sand"'
#    g_query = 'substrate == "gravel"'
#    b_query = 'substrate == "boulders"'    
    
    #25% ranges
    p_25 = get_center(cont_df,var_df,ent_df,homo_df,'percentile_25')
    p_50 = get_center(cont_df,var_df,ent_df,homo_df,'percentile_50')
    p_75 = get_center(cont_df,var_df,ent_df,homo_df,'percentile_75')
    #df = p_50
    for df in [p_25,p_50,p_75]:
        
        if df.equals(p_25):
            metric = 'percentile25'
        elif df.equals(p_50):
            metric= 'percentile50'
        elif df.equals(p_75):
            metric = 'percentile75'
        outFile = r"C:\workspace\GLCM\output\least_sqares_classification\R01346_R01347_"+metric+"_Sed_Class_4_variable.tif"
    
        
        #======================================================
        ## inputs
        w = [1,1,1] #weightings - leave at 1 unless you have any preference for 1 input variable over another. 
        
        # calibration matrix consisting of N rows (substrates, e.g. 4 (null, sand, gravel, boulders)) and M columns (classifiers - e.g M=3 for entropy, homo, and glcm variance)
        # so calib contains the means of those classifier variables per substrate
        # note that M can be more than 3
        
        calib = np.asarray(df.values,dtype='float')
        #calib=calib[:,:4]
        vec1 = ent_data.flatten()#flattened array of homogeneity values from a given sample (sidescan)
        vec2 = cont_data.flatten()#flattened array of entropy values
        vec3 = var_data.flatten()#flattened array of GLCM variance values
        vec4 = 1- homo_data.flatten()
        
        vec1[np.isnan(vec1)] = 0; vec2[np.isnan(vec2)] = 0; vec3[np.isnan(vec3)] = 0;vec4[np.isnan(vec4)]=0
        vec1[np.isinf(vec1)] = 0; vec2[np.isinf(vec2)] = 0; vec3[np.isinf(vec3)] = 0;vec4[np.isinf(vec4)]=0
        ind = np.nonzero(vec3)[0]
        # =============== 
        # classify!
        # pre-allocate arrays
        prc_sand = np.zeros(np.shape(vec1))*np.nan
        prc_gravel = np.zeros(np.shape(vec1))*np.nan
        prc_rock = np.zeros(np.shape(vec1))*np.nan
        ss_resid = np.zeros(np.shape(vec1))*np.nan # residual norm
        
        
        # classify 
        for k in xrange(len(ind)):
              prc_sand[ind[k]], prc_gravel[ind[k]], prc_rock[ind[k]], ss_resid[ind[k]] = get_class(calib.T,(vec1[ind[k]],vec2[ind[k]],vec3[ind[k]],vec4[ind[k]]),w)#
        
        # now reshape the arrays
        
        # =============== reshape
        Ny, Nx = np.shape(ent_data)
        prc_sand = np.reshape(prc_sand,(Ny, Nx))
        prc_gravel = np.reshape(prc_gravel,(Ny, Nx))
        prc_rock = np.reshape(prc_rock,(Ny, Nx))
        ss_resid = np.reshape(ss_resid,(Ny, Nx))
        
        import matplotlib.pyplot as plt
        
        plt.imshow(ss_resid);plt.colorbar();plt.show()
        
        # =============== define confidence metric
        sand_conf = prc_sand*(1-prc_rock)*(1-prc_gravel)
        rock_conf = prc_rock*(1-prc_sand)*(1-prc_gravel)
        gravel_conf = prc_gravel*(1-prc_sand)*(1-prc_rock)
        
        plt.imshow(sand_conf);plt.colorbar();plt.show()
        
          
        sed_df = pd.DataFrame({'prc_sand':prc_sand.flatten(),'sand_conf':sand_conf.flatten(),'prc_gravel':prc_gravel.flatten(),
        'gravel_conf':gravel_conf.flatten(),'prc_rock':prc_rock.flatten(),'rock_conf':rock_conf.flatten()})
        
        sed_df = sed_df[['prc_sand','sand_conf','prc_gravel','gravel_conf','prc_rock','rock_conf']]
        
        sed_df['sedclass'] = sed_df.apply(lambda row: make_class(row),axis=1)
        
        sed_class = np.reshape(np.asarray(sed_df['sedclass'],dtype='float'),np.shape(sand_conf))
        plt.imshow(sed_class);plt.colorbar();plt.show()
        
        CreateRaster(sed_class,gt,outFile)