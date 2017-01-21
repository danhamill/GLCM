# -*- coding: utf-8 -*-
"""
Created on Wed Jan 04 10:04:22 2017

@author: dan
"""
from scikits.bootstrap import bootstrap as boot
from rasterstats import zonal_stats
import pandas as pd
from osgeo import osr, gdal,ogr
import numpy as np
import os
from skimage.feature import greycomatrix, greycoprops
from skimage.segmentation import slic
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pytablewriter



####################################################################################################################
def entropy_calc(glcm):
    with np.errstate(divide='ignore', invalid='ignore'):
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

def crop_toseg(mask, im):
    true_points = np.argwhere(mask)
    top_left = true_points.min(axis=0)
    # take the largest points and use them as the bottom right of your crop
    bottom_right = true_points.max(axis=0)
    cmask = mask[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
    cim = im[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
    cmask = cmask.astype(bool)
    nrows, ncols = np.shape(cmask)
    
    maxarea = (0, [])
    
    arr = np.asarray(cmask, dtype='int')
    wid = np.zeros(dtype=int, shape=arr.shape)
    hght = np.zeros(dtype=int, shape=arr.shape)
    
    for row in xrange(nrows):
        for col in xrange(ncols):
            if arr[row][col] == 0:
                continue
    
            if row == 0:
                hght[row][col] = 0
            else:
                hght[row][col] = hght[row-1][col]+1
    
            if col == 0:
                wid[row][col] = 0
            else:
                wid[row][col] = wid[row][col-1]+1
            min_w = wid[row][col]
    
            for delta_hght in xrange(hght[row][col]):
                min_w = np.min([min_w, wid[row-delta_hght][col]])
                areanow = (delta_hght+1)*min_w
                if areanow > maxarea[0]:
                    maxarea = (areanow, [(row-delta_hght, col-min_w+1, row, col)])
    
    t = np.squeeze(maxarea[1])
    
    min_row = t[0]; max_row = t[2]
    min_col = t[1]; max_col = t[3]
         
    cim = cim[min_row:max_row, min_col:max_col]
       
    return cmask,cim 
    
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
        count = np.count_nonzero(mask)
        im_count =  np.count_nonzero(im[segments_slic == k])
        
        #Attempt to weed out no data super pixels
        if im_count > 0.75*count:
    
            # compute GLCM using 3 distances over 4 angles
            glcm = greycomatrix(cim, [5], [0], 256, symmetric=True, normed=True)
    
            #populate masks for 4 glcm variables
            ent[segments_slic == k] = entropy_calc(glcm)[0, 0]
            var[segments_slic == k] = mean_var(glcm)[0,0]
            homo[segments_slic == k] = greycoprops(glcm, 'homogeneity')[0, 0]
        else:
           #populate masks for 4 glcm variables
            ent[segments_slic == k] = 0
            var[segments_slic == k] = 0
            homo[segments_slic == k] = 0

    
    #mask out no data portions of the input image   
    ent[ent>15]= np.nan
    ent[np.isnan(m)] = np.nan  
    var[np.isnan(m)] = np.nan  
    homo[np.isnan(m)] = np.nan  
    return (ent, var, homo)

####################################################################################################################
    
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

def lsq_read_raster(raster):
    ds = gdal.Open(raster)
    data = ds.GetRasterBand(1).ReadAsArray()
    data[data==-99] = np.nan
    gt = ds.GetGeoTransform()
    return data, gt
    
def CreateRaster(xx,yy,std,gt,proj,driverName,outFile):  
    '''
    Exports data to GTiff Raster
    '''
    std = np.squeeze(std)
    std[std == 0] = -99
    std[np.isinf(std)] = -99
    std[np.isnan(std)] = -99
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
def lsq_CreateRaster(sed_class,gt,outFile):  
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
      
def make_glcm_raster(ent,var,homo,v1,v2,v3):
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(26949)
    
    CreateRaster(xx,yy,ent,gt,proj,'GTiff',v1)
    CreateRaster(xx,yy,var,gt,proj,'GTiff',v2)
    CreateRaster(xx,yy,homo,gt,proj,'GTiff',v3)
        
####################################################################################################################

   
def agg_distributions(stats,in_shp,metric):
    #Lets get get the substrate to sort lists
    a = get_subs(in_shp)

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


    
def plot_distributions(merge_dist,iter_start):
    #legend stuff
    blue = mpatches.Patch(color='blue',label='Sand')
    green = mpatches.Patch(color='green',label='Gravel')
    red = mpatches.Patch(color='red',label='Boulders')
    
    fig, [ax,ax1,ax2] = plt.subplots(nrows=3)
    merge_dist.groupby('sedclass')['Entropy'].plot.hist(ax=ax,bins=50,normed=True)    
    ax.set_xlabel('Entropy')
    ax.legend(loc=9,handles=[blue,green,red],ncol=3,columnspacing=1, fontsize=8)
    merge_dist.groupby('sedclass')['Homogeneity'].plot.hist(ax=ax1,bins=50,normed=True)    
    ax1.set_xlabel('Homogeneity')
    ax1.legend(loc=9,handles=[blue,green,red],ncol=3,columnspacing=1, fontsize=8)
    merge_dist.groupby('sedclass')['Variance'].plot.hist(ax=ax2,bins=50,normed=True)         
    ax2.set_xlabel('GLCM Variance')
    ax2.legend(loc=9,handles=[blue,green,red],ncol=3,columnspacing=1, fontsize=8)
    plt.tight_layout()
    title = str(iter_start) + ' Merged Distributions'
    plt.suptitle(title)
    plt.show()
    oName = root + os.sep + 'GLCM_aggregrated_distributions.png'
    plt.savefig(oName,dpi=600)   
    plt.close()


def get_zstats(ent_raster,var_raster,homo_raster,in_shp,fnames):        
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
    oName = root + os.sep + k + "_aggregraded_distributions.csv"
    fnames.append(oName)
    agg_dist.to_csv(oName,sep=',',index=False)
    return fnames     
    
def merge_zonal_stats(fnames, root):
    a = []
    variable = 'entropy'
    df1 = pd.read_csv(fnames[0],sep=',')
    df2 = pd.read_csv(fnames[3],sep=',')
    df3 = pd.read_csv(fnames[6],sep=',')
    oName = root + os.sep + variable + "_zonal_stats_merged.csv"
    merge= pd.concat([df1,df2,df3])
    merge.to_csv(oName,sep=',',index=False)
    a.append(oName)
    variable = 'variance'
    df1 = pd.read_csv(fnames[1],sep=',')
    df2 = pd.read_csv(fnames[4],sep=',')
    df3 = pd.read_csv(fnames[7],sep=',')
    oName = root + os.sep + variable + "_zonal_stats_merged.csv"
    merge= pd.concat([df1,df2,df3])
    merge.to_csv(oName,sep=',',index=False)
    a.append(oName)
    variable = 'homogeneity'
    df1 = pd.read_csv(fnames[2],sep=',')
    df2 = pd.read_csv(fnames[5],sep=',')
    df3 = pd.read_csv(fnames[8],sep=',')
    oName = root + os.sep + variable + "_zonal_stats_merged.csv"
    merge= pd.concat([df1,df2,df3])
    merge.to_csv(oName,sep=',',index=False)    
    a.append(oName)
    return a
    
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

def get_subs(shp):
    ds = ogr.Open(shp)
    lyr = ds.GetLayer(0)
    a=[]
    for row in lyr:
        a.append(row.substrate)
    lyr.ResetReading()
    del ds
    return a

def seg_area(segments_slic,ss_raster,k):
    im = read_raster(ss_raster)[0]
    test = segments_slic.copy()
    test[np.isnan(im)] = -99
    unique, counts = np.unique(test[~np.isnan(im)], return_counts=True)
    return np.average(counts)           
 
def plot_area(segments_slic,ss_raster,k,root):
    im = read_raster(ss_raster)[0]
    test = segments_slic.copy()
    test[np.isnan(im)] = -99
    unique, counts = np.unique(test[~np.isnan(im)], return_counts=True)
    df = pd.DataFrame({'unique':unique, 'counts':counts})
    fig,ax = plt.subplots()
    df.plot.bar(ax=ax,x=df['unique'],y='counts')
    ax.set_ylabel('Cell Counts')
    ax.set_xlabel('Unique slic segmentation label')
    oName = root + os.sep + "slic_segmentation_area" + os.sep + k + "_area_distributions.png"
    plt.tight_layout()
    plt.savefig(oName,dpi=600)
    plt.close()
        
if __name__ == '__main__':
    
    root = r'C:\workspace\GLCM\slic_output\blob1'
    
    #set up new directory
    if not os.path.exists(root + os.sep + 'slic_glcm_rasters'):
        os.mkdir(root + os.sep + 'slic_glcm_rasters')
        os.mkdir(root + os.sep + 'slic_glcm_rasters' + os.sep + '2014_04')
        os.mkdir(root + os.sep + 'slic_glcm_rasters' + os.sep + '2014_09')
        os.mkdir(root + os.sep + 'slic_lsq_rasters' )
        os.mkdir(root + os.sep + 'slic_segmentation_area')
        
        
    
    #Input sidescan rasters
    ss_dict = {'R01346':r"C:\workspace\Merged_SS\window_analysis\10_percent_shift\raster\ss_50_rasterclipped.tif",
                'R01765':r"C:\workspace\Merged_SS\raster\2014_09\ss_2014_09_R01765_raster.tif",
                'R01767':r"C:\workspace\Merged_SS\raster\2014_09\ss_2014_09_R01767_raster.tif"}
                             
    #Output Rasters
    ent_dict = {'R01346':root + os.sep + r"slic_glcm_rasters\2014_04\R01346_R01347_slic_entropy.tif",
                'R01765':root + os.sep + r"slic_glcm_rasters\2014_09\R01765_slic_entropy.tif",
                'R01767':root + os.sep + r"slic_glcm_rasters\2014_09\R01767_slic_entropy.tif"}
                
    var_dict = {'R01346':root + os.sep + r"slic_glcm_rasters\2014_04\R01346_R01347_slic_var.tif",
                'R01765':root + os.sep + r"slic_glcm_rasters\2014_09\R01765_3_slic_var.tif",
                'R01767':root + os.sep + r"slic_glcm_rasters\2014_09\R01767_3_slic_var.tif"}       
    
    homo_dict = {'R01346':root + os.sep + r"slic_glcm_rasters\2014_04\R01346_R01347_slic_homo.tif",
                 'R01765':root + os.sep + r"slic_glcm_rasters\2014_09\R01765_slic_homo.tif",
                 'R01767':root + os.sep + r"slic_glcm_rasters\2014_09\R01767_slic_homo.tif"}  
    
    shp_dict = {'R01346':r"C:\workspace\Merged_SS\window_analysis\shapefiles\tex_seg_800_3class.shp",
                'R01765':r"C:\workspace\Merged_SS\window_analysis\shapefiles\R01765.shp",
                'R01767':r"C:\workspace\Merged_SS\window_analysis\shapefiles\tex_seg_2014_09_67_3class.shp"}  
    fnames = []
    
    iter_start = [int(1000),int(1000), int(250)]
    fig_sizes = [(5,6),(4,8),(4,8)]
    n = 0
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
        print 'Now calculating n_segments for slic segments for %s...' %(k,)
        segs = int(iter_start[n])
        segments_slic = slic(im, n_segments=segs, compactness=.1,slic_zero=True)
        
        
        im = read_raster(ss_raster)[0]
        print 'Now calculating GLCM metrics for %s...' %(k,)
        #Calculate GLCM metrics for slic segments
        ent,var,homo = glcm_calc(im,segments_slic)
        
        print 'Now making rasters...'
        #Write GLCM rasters to file
        make_glcm_raster(ent,var,homo,v1,v2,v3)
        
        print 'Aggregrating Distributions...'
        #Aggregrate distributions and save to file
        fnames = get_zstats(ent_raster,var_raster,homo_raster,in_shp,fnames)
       
        n += 1
        del (k,v), (k1,v1), (k2,v2), (k3,v3), (k4,v4),xx,yy, ent_raster,var_raster,homo_raster,in_shp,im, segments_slic,ent,var,homo,gt,segs
    del n    

    #Merge GLCM distributions for plotting
    df1 = pd.read_csv(fnames[0],sep=',')  
    df2 = pd.read_csv(fnames[1],sep=',')  
    df3 = pd.read_csv(fnames[2],sep=',')
    
    merge_dist = pd.concat([df1,pd.concat([df2,df3])])
    del df1,df2,df3
    oName = root + os.sep + "merged_aggregraded_distributions.csv"
    merge_dist.to_csv(oName,sep=',',index=False)    
    plot_distributions(merge_dist,iter_start)
    del fnames, merge_dist,iter_start
    
#    print 'Calculating zonal statistics for GLCM rasters...'
#    #calculate zonal statisitcs and write to file
#    fnames = []
#    for (k,v), (k1,v1), (k2,v2), (k3,v3), (k4,v4) in zip(ss_dict.items(),ent_dict.items(),var_dict.items(), homo_dict.items(),shp_dict.items()):
#        ss_raster = v
#        ent_raster = v1
#        var_raster = v2
#        homo_raster= v3
#        in_shp = v4
#        
#        ds = ogr.Open(in_shp)
#        lyr = ds.GetLayer(0)
#        a=[]
#        for row in lyr:
#            a.append(row.substrate)
#        lyr.ResetReading()
#        del ds
#        
#        variable = 'entropy'
#        oName = root + os.sep  + k + "_" + variable + "_zonalstats.csv" 
#        ent_stats = zonal_stats(in_shp, ent_raster, stats=['min','mean','max','median','std','count','percentile_25','percentile_50','percentile_75'])
#        df = pd.DataFrame(ent_stats)
#        df['substrate'] = a
#        df.to_csv(oName,sep=',',index=False)
#        fnames.append(oName)
#        
#        variable = 'variance'
#        oName = root + os.sep  + k + "_" + variable + "_zonalstats.csv" 
#        var_stats = zonal_stats(in_shp, var_raster, stats=['min','mean','max','median','std','count','percentile_25','percentile_50','percentile_75'])
#        df = pd.DataFrame(var_stats)
#        df['substrate'] = a      
#        df.to_csv(oName,sep=',',index=False)
#        fnames.append(oName)
#        
#        variable = 'homogeneity'
#        oName = root + os.sep + k + "_" + variable + "_zonalstats.csv" 
#        homo_stats = zonal_stats(in_shp, homo_raster, stats=['min','mean','max','median','std','count','percentile_25','percentile_50','percentile_75'])
#        df = pd.DataFrame(homo_stats)
#        df['substrate'] = a 
#        df.to_csv(oName,sep=',',index=False)
#        fnames.append(oName)
#        
#        del (k,v), (k1,v1), (k2,v2), (k3,v3), (k4,v4),ss_raster,ent_raster,var_raster,homo_raster,in_shp,a,df, oName, variable
#    
#    #merge zonal stats csv files
#    fnames = merge_zonal_stats(fnames,root)
#    
#    try:
#        fnames
#    except:
#        fnames = [root + os.sep + "entropy_zonal_stats_merged.csv",                 
#                  root + os.sep + "variance_zonal_stats_merged.csv",
#                  root + os.sep + "homogeneity_zonal_stats_merged.csv"]
#
#    
#    #Begin of lsq classifications
#    df1 = pd.read_csv(fnames[2],sep=',')
#    df2 = pd.read_csv(fnames[0],sep=',')
#    df3 = pd.read_csv(fnames[1],sep=',')
#    del fnames
#    df1.rename(columns={'max':'homo_max', 'mean':'homo_mean', 'median':'homo_median','min':'homo_min','percentile_25':'homo_25','percentile_50':'homo_50', 'percentile_75':'homo_75','std':'homo_std'},inplace=True)   
#    df2.rename(columns={'max':'entropy_max', 'mean':'entropy_mean', 'median':'entropy_median','min':'entropy_min','percentile_25':'entropy_25','percentile_50':'entropy_50', 'percentile_75':'entropy_75','std':'entropy_std'},inplace=True)   
#    df3.rename(columns={'max':'var_max', 'mean':'var_mean', 'median':'var_median','min':'var_min','percentile_25':'var_25','percentile_50':'var_50', 'percentile_75':'var_75','std':'var_std'},inplace=True)   
#    
#    
#    merge =df1.merge(df2,left_index=True, right_index=True, how='left')
#    merge = merge.merge(df3,left_index=True, right_index=True, how='left' )
#    merge = merge[['homo_median','entropy_median','var_median','substrate']].dropna()
#    merge.rename(columns={'substrate_x':'substrate'},inplace=True)
#    del df1,df2,df3
#    grouped = merge.groupby('substrate')
#    sand = grouped.get_group('sand')
#    gravel = grouped.get_group('gravel')
#    boulders = grouped.get_group('boulders')
#    del merge
#    
#    print 'Calculating calibrations metrics for lsq classifications...'
#    calib_df = pd.DataFrame(columns=['ent','homo','var'], index=['sand','gravel','boulders'])
#
#    calib_df.loc['sand'] = pd.Series({'homo':1- np.average(boot.ci(sand['homo_median'],np.median,alpha=0.05)) ,
#                                    'ent':np.average(boot.ci(sand['entropy_median'],np.median,alpha=0.05)) ,
#                                    'var': np.average(boot.ci(sand['var_median'],np.median,alpha=0.05))})
#    calib_df.loc['gravel'] = pd.Series({'homo':1- np.average(boot.ci(gravel['homo_median'],np.median,alpha=0.05)) ,
#                                    'ent':np.average(boot.ci(gravel['entropy_median'],np.median,alpha=0.05)) ,
#                                    'var': np.average(boot.ci(gravel['var_median'],np.median,alpha=0.05))})
#    calib_df.loc['boulders'] = pd.Series({'homo':1- np.average(boot.ci(boulders['homo_median'],np.median,alpha=0.05)) ,
#                                    'ent':np.average(boot.ci(boulders['entropy_median'],np.median,alpha=0.05)) ,
#                                    'var': np.average(boot.ci(boulders['var_median'],np.median,alpha=0.05))})
#    
#    del sand, gravel, boulders
#    
#    fnames = []
#    for (k,v), (k1,v1), (k2,v2) in zip(ent_dict.items(),var_dict.items(), homo_dict.items()):
#        print 'Now making %s LSQ classificatons...' %(k,)
#        ent_raster = v
#        var_raster = v1
#        homo_raster= v2
#
#        #Get the data
#        ent_data,gt = lsq_read_raster(ent_raster)
#        var_data = lsq_read_raster(var_raster)[0]
#        homo_data = lsq_read_raster(homo_raster)[0]
#        df = pd.DataFrame({'ent':ent_data.flatten(),'var':var_data.flatten(),'homo':homo_data.flatten()})
#    
#        outFile = root + os.sep + "slic_lsq_rasters" + os.sep + k +"_median_Sed_Class_3_variable.tif"
#        fnames.append(outFile)
#        #======================================================
#        ## inputs
#        w = [0.1,0.7,0.2] #weightings - leave at 1 unless you have any preference for 1 input variable over another. 
#        
#        calib = np.asarray(calib_df.values,dtype='float')
#        vec1 = ent_data.flatten()#flattened array of homogeneity values from a given sample (sidescan)
#        vec2 = 1-homo_data.flatten()#flattened array of entropy values
#        vec3 = var_data.flatten()#flattened array of GLCM variance values
#        ##vec4 = #??
#        
#        vec1[np.isnan(vec1)] = 0; vec2[np.isnan(vec2)] = 0; vec3[np.isnan(vec3)] = 0
#        vec1[np.isinf(vec1)] = 0; vec2[np.isinf(vec2)] = 0; vec3[np.isinf(vec3)] = 0
#        unique, ind = np.unique(vec1,return_index=True)
#        #ind = np.nonzero(vec1)[0]
#        # =============== 
#        # classify!
#        # pre-allocate arrays
#        prc_sand = np.zeros(np.shape(vec1))*np.nan
#        prc_gravel = np.zeros(np.shape(vec1))*np.nan
#        prc_rock = np.zeros(np.shape(vec1))*np.nan
#        ss_resid = np.zeros(np.shape(vec1))*np.nan # residual norm
#        
#        
#        # classify 
#        for k in xrange(len(ind[1:])):
#              prc_sand[vec1==unique[k]], prc_gravel[vec1==unique[k]], prc_rock[vec1==unique[k]], ss_resid[vec1==unique[k]] = get_class(calib.T,(vec1[ind[k]],vec2[ind[k]],vec3[ind[k]]),w)
#        
#        # now reshape the arrays
#        
#        # =============== reshape
#        Ny, Nx = np.shape(ent_data)
#        prc_sand = np.reshape(prc_sand,(Ny, Nx))
#        prc_gravel = np.reshape(prc_gravel,(Ny, Nx))
#        prc_rock = np.reshape(prc_rock,(Ny, Nx))
#        ss_resid = np.reshape(ss_resid,(Ny, Nx))
#        
#        # =============== define confidence metric
#        sand_conf = prc_sand*(1-prc_rock)*(1-prc_gravel)
#        rock_conf = prc_rock*(1-prc_sand)*(1-prc_gravel)
#        gravel_conf = prc_gravel*(1-prc_sand)*(1-prc_rock)
#          
#        sed_df = pd.DataFrame({'prc_sand':prc_sand.flatten(),'sand_conf':sand_conf.flatten(),'prc_gravel':prc_gravel.flatten(),
#        'gravel_conf':gravel_conf.flatten(),'prc_rock':prc_rock.flatten(),'rock_conf':rock_conf.flatten()})
#        
#        sed_df = sed_df[['prc_sand','sand_conf','prc_gravel','gravel_conf','prc_rock','rock_conf']]
#        
#        sed_df['sedclass'] = sed_df.apply(lambda row: make_class(row),axis=1)
#
#        
#        sed_class = np.reshape(np.asarray(sed_df['sedclass'],dtype='float'),np.shape(sand_conf))
#        
#        lsq_CreateRaster(sed_class,gt,outFile)
#
#        del k, k1, k2, v1, v, v2, sed_class, sed_df, sand_conf, rock_conf, gravel_conf,prc_sand,prc_gravel,prc_rock,ss_resid,vec1,vec2,vec3
#        del outFile,homo_data,var_data,ent_data,ind,gt,Nx,Ny,ent_raster,homo_raster,var_raster,unique
#        
#    #convert fnames to dict for classification results
#    lsq_dict = {'R01346':fnames[1],'R01765':fnames[2],'R01767':fnames[0]}
#    n=0
#    for (k,v), (k1,v1)in zip(lsq_dict.items(),shp_dict.items()[0:1]):
#        raster = v
#        shp = v1
#        subs = get_subs(shp)
#        stats = zonal_stats(shp, raster, categorical=True, nodata=-99)
#        if n == 0:
#            merge_subs = subs
#            merge_df = pd.DataFrame(stats)
#        else:
#            merge_subs.extend(subs)
#            merge_df = pd.concat([merge_df,pd.DataFrame(stats)])
#        del stats, shp,raster,subs,(k,v), (k1,v1)
#        n += 1
#    del n
#    
#    merge_df['substrate'] = merge_subs
#    
#    try:
#        merge_df.rename(columns = {0.0:'null',1.0:'Sand',2.0:'Gravel',3.0:'Boulders'},inplace=True)
#        merge_df = merge_df[['null','Sand','Gravel','Boulders','substrate']]   
#        pvt = pd.pivot_table(merge_df, index=['substrate'],values=['null','Sand','Gravel','Boulders'],aggfunc=np.nansum)
#    except:
#        merge_df.rename(columns = {1.0:'Sand',2.0:'Gravel',3.0:'Boulders'},inplace=True)
#        merge_df = merge_df[['Sand','Gravel','Boulders','substrate']]
#        pvt = pd.pivot_table(merge_df, index=['substrate'],values=['Sand','Gravel','Boulders'],aggfunc=np.nansum)
#    
#    del merge_df
#    #Percentage classification table
#    class_df = pvt.div(pvt.sum(axis=1), axis=0)
#
#    writer = pytablewriter.MarkdownTableWriter()
#    writer.table_name = "Non-negative Least Squares Classification Results"
#    writer.header_list = list(class_df.columns.values)
#    writer.value_matrix = class_df.values.tolist()
#    writer.write_table()
#    
#    writer = pytablewriter.MarkdownTableWriter()
#    writer.table_name = "Calibration Dataframe"
#    writer.header_list = list(calib_df.reset_index().columns.values)
#    writer.value_matrix = calib_df.reset_index().values.tolist()
#    writer.write_table()   