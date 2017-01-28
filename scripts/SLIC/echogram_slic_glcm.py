# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 11:31:00 2017

@author: dan
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
from skimage.segmentation import slic, mark_boundaries
from scipy.io import loadmat
import numpy as np
from joblib import Parallel, delayed, cpu_count

import PyHum.utils as humutils
import PyHum.getxy as getxy
import PyHum.io as io

import pyresample
import pyproj
from osgeo import gdal,osr

from skimage.feature import greycomatrix, greycoprops
import time



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
                areanow = (delta_hght+5)*min_w
                if areanow > maxarea[0]:
                    maxarea = (areanow, [(row-delta_hght, col-min_w+5, row, col)])
    
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
    
            # compute GLCM using 1 distances over 1 angles
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
    ent[np.isnan(m)] = np.nan  
    var[np.isnan(m)] = np.nan  
    homo[np.isnan(m)] = np.nan  
    return (ent, var, homo)
    
    
    
# =========================================================
def xyfunc(e,n,yvec,d,t,extent):
    return getxy.GetXY(e, n, yvec, d, t, extent).getdat2()

# =========================================================
def getXY(e,n,yvec,d,t,extent):
    print "getting point cloud ..."

    if os.name=='nt':
        o = Parallel(n_jobs = cpu_count(), verbose=0)(delayed(xyfunc)(e[k], n[k], yvec, d[k], t[k], extent) for k in xrange(len(n)))

        #eating, northing, distance to sonar, depth, heading
        X, Y = zip(*o)

    else:
        X = []; Y = [];
        for k in xrange(len(n)):
            out1,out2 = xyfunc(e[k], n[k], yvec, d[k], t[k], extent)
            X.append(out1); Y.append(out2)

    # merge flatten and stack
    X = np.asarray(X,'float').T
    X = X.flatten()

    # merge flatten and stack
    Y = np.asarray(Y,'float').T
    Y = Y.flatten()

    return X, Y

# =========================================================
def CreateRaster(datm,gt,proj,cols,rows,driverName,outFile):  
     '''
     Exports data to GTiff Raster
     '''
     datm = np.squeeze(datm)
     datm[np.isnan(datm)] = -99
     driver = gdal.GetDriverByName(driverName)
     ds = driver.Create( outFile, cols, rows, 1, gdal.GDT_Float32)        
     if proj is not None:  
          ds.SetProjection(proj.ExportToWkt()) 
     ds.SetGeoTransform(gt)
     ss_band = ds.GetRasterBand(1)
     ss_band.WriteArray(datm)
     ss_band.SetNoDataValue(-99)
     ss_band.FlushCache()
     ss_band.ComputeStatistics(False)
     del ds
# =========================================================
def getmesh(minX, maxX, minY, maxY, res):
    
    complete=0
    while complete==0:
        try:
            grid_x, grid_y = np.meshgrid( np.arange(minX, maxX, res), np.arange(minY, maxY, res) )
            if 'grid_x' in locals(): 
                complete=1 
        except:
            print "memory error: trying grid resolution of %s" % (str(res*2))
            res = res*2
            
    return grid_x, grid_y, res    
# =========================================================     
def get_raster_size(minx, miny, maxx, maxy, cell_width, cell_height):
     """
     Determine the number of rows/columns given the bounds of the point data and the desired cell size
     """
     cols = int((maxx - minx) / cell_width)
     rows = int((maxy - miny) / abs(cell_height))
     return cols, rows
     
# =========================================================
def get_grid(mode, orig_def, targ_def, merge, influence, minX, maxX, minY, maxY, res, nn, sigmas, eps, shape, numstdevs, trans, humlon, humlat):

     if mode==1:

         wf = None
         
         complete=0
         while complete==0:
             try:
                 try:
                     dat = pyresample.kd_tree.resample_nearest(orig_def, merge.flatten(), targ_def, radius_of_influence=res*20, fill_value=None, nprocs = cpu_count()-2) 
                 except:
                     dat = pyresample.kd_tree.resample_nearest(orig_def, merge.flatten(), targ_def, radius_of_influence=res*20, fill_value=None, nprocs = 1)                  
                 if 'dat' in locals(): 
                     complete=1 
             except:
                     print 'Something went wrong with resampling...'                 


     dat = dat.reshape(shape)

     return dat, res

# =========================================================
def get_griddefs(minX, maxX, minY, maxY, res, humlon, humlat, trans):  
    
     complete=0
     while complete==0:
         try:
             grid_x, grid_y, res = getmesh(minX, maxX, minY, maxY, res)
             longrid, latgrid = trans(grid_x, grid_y, inverse=True)
             shape = np.shape(grid_x)

             targ_def = pyresample.geometry.SwathDefinition(lons=longrid.flatten(), lats=latgrid.flatten())
             del longrid, latgrid

             orig_def = pyresample.geometry.SwathDefinition(lons=humlon.flatten(), lats=humlat.flatten())
             if 'orig_def' in locals(): 
                 complete=1 
         except:
             print "memory error: trying grid resolution of %s" % (str(res*2))
             res = res*2
                         
     return orig_def, targ_def, grid_x, grid_y, res, shape     
# modify paths below



if __name__ == '__main__':
    
    humfile = r"C:\workspace\Reach_4a\2014_09\R01767\R01767.DAT"
    sonpath = r"C:\workspace\Reach_4a\2014_09\R01767"
    # if son path name supplied has no separator at end, put one on
    if sonpath[-1]!=os.sep:
        sonpath = sonpath + os.sep

    base = humfile.split('.DAT') # get base of file name for output
    base = base[0].split(os.sep)[-1]
     # start timer
    if os.name=='posix': # true if linux/mac or cygwin on windows
       start = time.time()
    else: # windows
       start = time.clock()

    # remove underscores, negatives and spaces from basename
    base = humutils.strip_base(base)    

    meta = loadmat(os.path.normpath(os.path.join(sonpath,base+'meta.mat')))

    ### port
    print "processing port side ..."
    # load memory mapped scan ... port
    shape_port = np.squeeze(meta['shape_port'])
    if shape_port!='':

        if os.path.isfile(os.path.normpath(os.path.join(sonpath,base+'_data_port_lar.dat'))):
            port_fp = io.get_mmap_data(sonpath, base, '_data_port_lar.dat', 'float32', tuple(shape_port))            
        else:
            port_fp = io.get_mmap_data(sonpath, base, '_data_port_la.dat', 'float32', tuple(shape_port))
             

    ### star
    print "processing starboard side ..."
    # load memory mapped scan ... port
    shape_star = np.squeeze(loadmat(sonpath+base+'meta.mat')['shape_star'])
    if shape_star!='':
        if os.path.isfile(os.path.normpath(os.path.join(sonpath,base+'_data_star_lar.dat'))):
            star_fp = io.get_mmap_data(sonpath, base, '_data_star_lar.dat', 'float32', tuple(shape_star))
        else:
            star_fp = io.get_mmap_data(sonpath, base, '_data_star_la.dat', 'float32', tuple(shape_star))


    if len(shape_star)>2:
        shape = shape_port.copy()
        shape[1] = shape_port[1] + shape_star[1]
    else:
        shape = []
        shape.append(1)
        shape.append(shape_port[0])
        shape.append(shape_port[1])
        shape[1] = shape_port[0] + shape_star[0]

    #work on the entire scan
    im = np.vstack((np.flipud(np.hstack(port_fp)), np.hstack(star_fp)))
    im[np.isnan(im)] = 0
    im = humutils.rescale(im,0,1)
   
    print 'Calculating SLIC super pixels...'
    
    #get SLIC superpixels
    segments_slic = slic(im, n_segments=int(im.shape[0]/5), compactness=.1)
    
    # get number pixels in scan line
    extent = int(np.shape(im)[0]/2)
    del im
    #Reload image for GLCM calculations 
    im = np.vstack((np.flipud(np.hstack(port_fp)), np.hstack(star_fp)))
    im[np.isnan(im)] = 0
    del port_fp, star_fp
    
    fig, ax = plt.subplots()
    ax.imshow(mark_boundaries(im, segments_slic))
    plt.savefig(os.path.normpath(os.path.join(sonpath,'slic_segmentaion.png')),dpi=600)
    plt.close()
    print "Fount %s unique segments" % str(len(np.unique(segments_slic)))
    
    
    S = np.vstack((np.flipud(segments_slic[:extent,:]), segments_slic[extent:,:]))
   

    e = np.squeeze(meta['e'])
    n = np.squeeze(meta['n'])

    # time varying gain
    tvg = ((8.5*10**-5)+(3/76923)+((8.5*10**-5)/4))*meta['c']

    # depth correction
    d = np.squeeze(((np.tan(np.radians(25)))*np.squeeze(meta['dep_m']))-(tvg))
    t = np.squeeze(meta['heading'])/(180/np.pi)



    pix_m = meta['pix_m']

    yvec = np.squeeze(np.linspace(np.squeeze(pix_m),extent*np.squeeze(pix_m),extent))
    print 'Georectifying...'
    X, Y  = getXY(e,n,yvec,np.squeeze(d),t,extent)
    del t, d, e, n

    X = X[np.where(np.logical_not(np.isnan(Y)))]
    merge = S.flatten()[np.where(np.logical_not(np.isnan(Y)))]
    Y = Y[np.where(np.logical_not(np.isnan(Y)))]

    Y = Y[np.where(np.logical_not(np.isnan(X)))]
    merge = merge.flatten()[np.where(np.logical_not(np.isnan(X)))]
    X = X[np.where(np.logical_not(np.isnan(X)))]

    X = X[np.where(np.logical_not(np.isnan(merge)))]
    Y = Y[np.where(np.logical_not(np.isnan(merge)))]
    merge = merge[np.where(np.logical_not(np.isnan(merge)))]

    # plot to check
    #plt.scatter(X[::20],Y[::20],10,merge[::20], linewidth=0)

    print "writing point cloud"
    ## write raw bs to file
    outfile = os.path.normpath(os.path.join(sonpath,'x_y_slicsegmentnumber.asc'))

    np.savetxt(outfile, np.hstack((humutils.ascol(X.flatten()),humutils.ascol(Y.flatten()), humutils.ascol(merge.flatten()) )) , fmt="%8.6f %8.6f %8.6f") 
  
    trans =  pyproj.Proj(init="epsg:26949")
    humlon, humlat = trans(X, Y, inverse=True)
    res = 0.25

    orig_def, targ_def, grid_x, grid_y, res, shape = get_griddefs(np.min(X), np.max(X), np.min(Y), np.max(Y), res, humlon, humlat, trans)

    grid_x = grid_x.astype('float32')
    grid_y = grid_y.astype('float32')
                                  
    sigmas = 1 #m
    eps = 2
    mode = 1
    
    print 'Now Gridding slic superpixels...'
    dat, res = get_grid(mode, orig_def, targ_def, merge, res*10, np.min(X), np.max(X), np.min(Y), np.max(Y), res, 64, sigmas, eps, shape, 4, trans, humlon, humlat)
    dat = dat.astype('float64')
    dat[dat==0] = np.nan
    dat[np.isinf(dat)] = np.nan
 
    datm = np.ma.masked_invalid(dat)
  
 
    c,r = get_raster_size(np.floor(np.min(grid_x)),np.floor(np.min(grid_y)),np.ceil(np.max(grid_x)),np.ceil(np.max(grid_y)),0.25,0.25)
    gt = [np.floor(np.min(grid_x)),0.25,0,np.ceil(np.max(grid_y)),0,-0.25]
    driverName= 'GTiff'    
  
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(26949)
    
    oFile = os.path.normpath(os.path.join(sonpath + base +'slic.tif'))
    print 'Now Making SLIC superpixel raster...'
    CreateRaster(np.flipud(datm), gt, proj,c,r,driverName,oFile)
 # ##################################################################################
 

    print 'Now calculating GLCM metrics for Echogram...'
    #Calculate GLCM metrics for echogram
    ent,var,homo = glcm_calc(im,segments_slic)
    
    ent = np.vstack((np.flipud(ent[:extent,:]), ent[extent:,:]))
    var = np.vstack((np.flipud(var[:extent,:]), var[extent:,:]))
    homo = np.vstack((np.flipud(homo[:extent,:]), homo[extent:,:]))
    
    
    print 'Now resampling entropy...'
    merge = ent.flatten()[np.where(np.logical_not(np.isnan(Y)))]
    merge = merge.flatten()[np.where(np.logical_not(np.isnan(X)))]
    merge = merge[np.where(np.logical_not(np.isnan(merge)))]
    dat, res = get_grid(mode, orig_def, targ_def, merge, res*10, np.min(X), np.max(X), np.min(Y), np.max(Y), res, 64, sigmas, eps, shape, 4, trans, humlon, humlat)
    
    dat[dat==0] = np.nan
    dat[np.isinf(dat)] = np.nan
 
    datm = np.ma.masked_invalid(dat)
    mask = datm.mask==True
    fOut = os.path.normpath(os.path.join(sonpath,'x_y_entropy_25cm.asc'))
    print 'Writing point cloud to file...'
    with open(fOut, 'wb')as f:
        np.savetxt(f, np.hstack((humutils.ascol(grid_x[mask==False].flatten()),humutils.ascol(grid_y[mask==False].flatten()),humutils.ascol(datm[mask==False].flatten()))),delimiter=' ', fmt="%8.6f %8.6f %1.6f")   
    f.close()
    
    oFile = os.path.normpath(os.path.join(sonpath + base +'slic_ent.tif'))
    print 'Now Making entropy superpixel raster...'
    CreateRaster(np.flipud(datm), gt, proj,c,r,driverName,oFile)
    
    print 'Now resampling variance...'
    merge = var.flatten()[np.where(np.logical_not(np.isnan(Y)))]
    merge = merge.flatten()[np.where(np.logical_not(np.isnan(X)))]
    merge = merge[np.where(np.logical_not(np.isnan(merge)))]
    dat, res = get_grid(mode, orig_def, targ_def, merge, res*10, np.min(X), np.max(X), np.min(Y), np.max(Y), res, 64, sigmas, eps, shape, 4, trans, humlon, humlat)
    
    dat[dat==0] = np.nan
    dat[np.isinf(dat)] = np.nan
 
    datm = np.ma.masked_invalid(dat)
    fOut = os.path.normpath(os.path.join(sonpath,'x_y_variance25cm.asc'))
    print 'Writing point cloud to file...'
    with open(fOut, 'wb')as f:
        np.savetxt(f, np.hstack((humutils.ascol(grid_x[mask==False].flatten()),humutils.ascol(grid_y[mask==False].flatten()),humutils.ascol(datm[mask==False].flatten()))),delimiter=' ', fmt="%8.6f %8.6f %1.6f")   
    f.close()
    oFile = os.path.normpath(os.path.join(sonpath + base +'slic_var.tif'))
    print 'Now Making variance superpixel raster...'
    CreateRaster(np.flipud(datm), gt, proj,c,r,driverName,oFile)
    
    print 'Now resampling homogeneity...'
    merge = homo.flatten()[np.where(np.logical_not(np.isnan(Y)))]
    merge = merge.flatten()[np.where(np.logical_not(np.isnan(X)))]
    merge = merge[np.where(np.logical_not(np.isnan(merge)))]
    dat, res = get_grid(mode, orig_def, targ_def, merge, res*10, np.min(X), np.max(X), np.min(Y), np.max(Y), res, 64, sigmas, eps, shape, 4, trans, humlon, humlat)
    
    dat[dat==0] = np.nan
    dat[np.isinf(dat)] = np.nan
 
    datm = np.ma.masked_invalid(dat)
    
    print 'Writing point cloud to file...'
    fOut = os.path.normpath(os.path.join(sonpath,'x_y_homo25cm.asc'))
    with open(fOut, 'wb')as f:
        np.savetxt(f, np.hstack((humutils.ascol(grid_x[mask==False].flatten()),humutils.ascol(grid_y[mask==False].flatten()),humutils.ascol(datm[mask==False].flatten()))),delimiter=' ', fmt="%8.6f %8.6f %1.6f")   
    f.close()    
    
    oFile = os.path.normpath(os.path.join(sonpath + base +'slic_homo.tif'))
    print 'Now Making homo superpixel raster...'
    CreateRaster(np.flipud(datm), gt, proj,c,r,driverName,oFile)
    if os.name=='posix': # true if linux/mac
       elapsed = (time.time() - start)
    else: # windows
       elapsed = (time.clock() - start)
    print "Processing took ", elapsed , "seconds to analyse"

    print "Done!"
 