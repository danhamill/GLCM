# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 10:56:18 2016

@author: dan
"""

import gdal, osr
import numpy as np
from scipy.interpolate import RectBivariateSpline
from numpy.lib.stride_tricks import as_strided as ast
import dask.array as da
from joblib import Parallel, delayed, cpu_count
import os
from skimage.feature import greycomatrix, greycoprops
import sys

def angle_converter(angle):
    if int(angle) ==  0:
        return 0
    if int(angle) == 45:
        return np.pi/4
    if int(angle) == 90:
        return np.pi/2
    if int(angle) == 135:
        return 0.75*np.pi
        
def im_resize(im,Nx,Ny):
    '''
    resize array by bivariate spline interpolation
    '''
    ny, nx = np.shape(im)
    xx = np.linspace(0,nx,Nx)
    yy = np.linspace(0,ny,Ny)
    
    try:
        im = da.from_array(im, chunks=1000)   #dask implementation
    except:
        pass
    try:
        newKernel = RectBivariateSpline(np.r_[:ny],np.r_[:nx],im)
        return newKernel(yy,xx)
    except:
        newKernel = RectBivariateSpline(np.r_[:ny],np.r_[:nx],im,ky=2)
        return newKernel(yy,xx)

def mean_var(P):
    (num_level, num_level2, num_dist, num_angle) = P.shape
    I, J = np.ogrid[0:num_level, 0:num_level]
    I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
    mean_i = np.apply_over_axes(np.sum, (I * P), axes=(0, 1))[0, 0]
    diff_i = I - np.apply_over_axes(np.sum, (I * P), axes=(0, 1))[0, 0]
    var_h = np.apply_over_axes(np.sum, (P * (diff_i) ** 2), axes=(0, 1))[0, 0]
    return mean_i, var_h 
    
def entropy_calc(glcm):
    horizontal_entropy = np.apply_over_axes(np.nansum,(np.log(glcm)*-1*glcm),axes=(0,1))[0,0]
    horizontal_entropy = np.asarray([[horizontal_entropy[0,0]]])
    return horizontal_entropy
    
def p_me(Z, win,dist,angle):
    '''
    loop to standard deviation
    '''
    if np.count_nonzero(Z) > 0.75*win**2: 
        glcm = greycomatrix(Z, [dist], [angle], 256, symmetric=True, normed=True)
        cont = greycoprops(glcm, 'contrast')
        diss = greycoprops(glcm, 'dissimilarity')
        homo = greycoprops(glcm, 'homogeneity')
        eng = greycoprops(glcm, 'energy')
        corr = greycoprops(glcm, 'correlation')
        ASM = greycoprops(glcm, 'ASM')
        entropy = entropy_calc(glcm)
        mean, var = mean_var(glcm)
        return (cont, diss, homo, eng, corr, ASM, entropy,mean,var)
    else:
        return (0,0,0,0,0,0,0,0,0)
        
        
def read_raster(in_raster):
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
    
def norm_shape(shap):
   '''
   Normalize numpy array shapes so they're always expressed as a tuple,
   even for one-dimensional shapes.
   '''
   try:
      i = int(shap)
      return (i,)
   except TypeError:
      # shape was not a number
      pass

   try:
      t = tuple(shap)
      return t
   except TypeError:
      # shape was not iterable
      pass

   raise TypeError('shape must be an int, or a tuple of ints')

def sliding_window(a, ws, ss = None, flatten = True):
    '''
    Return a sliding window over a in any number of dimensions
    '''      
    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)
    # convert ws, ss, and a.shape to numpy arrays
    ws = np.array(ws)
    ss = np.array(ss)
    shap = np.array(a.shape)
    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shap),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))
    
    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shap):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
     a.shape was %s and ws was %s' % (str(a.shape),str(ws)))
     
    # how many slices will there be in each dimension?
    newshape = norm_shape(((shap - ws) // ss) + 1)
    
    
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    
    
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    a = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return a
    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    dim = filter(lambda i : i != 1,dim)
    
    return a.reshape(dim), newshape
    
def CreateRaster(xx,yy,std,gt,proj,driverName,outFile):  
    '''
    Exports data to GTiff Raster
    '''
    std = np.squeeze(std)
    std[np.isinf(std)] = -99
    std[std>100] = -99
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
    
    
if __name__ == '__main__':  
    
    angle = 0#sys.argv[1]
    dist = 5#int(sys.argv[2])
    angle = angle_converter(angle)
    print 'Now working on %s angle and %ss distance...' %(str(angle), str(dist))      
    #Stuff to change
    win_sizes = [8,12,20,40,80]
    for win_size in win_sizes[1:2]:   
        in_raster = r"C:\workspace\Merged_SS\raster\2014_09\ss_2014_09_R01765_raster.tif"
        win = win_size
        meter = str(win/4)
        print 'Now working on %s meter grid...' %(meter,)
        contFile = r"C:\workspace\GLCM\output\new_glcm_rasters\2014_09_2" + os.sep + meter +os.sep+"R01765_" + meter + "_contrast.tif"
        dissFile = r"C:\workspace\GLCM\output\new_glcm_rasters\2014_09_2" + os.sep + meter +os.sep+"R01765_" + meter + "_diss.tif"
        homoFile = r"C:\workspace\GLCM\output\new_glcm_rasters\2014_09_2" + os.sep + meter +os.sep+"R01765_" + meter + "_homo.tif"
        energyFile = r"C:\workspace\GLCM\output\new_glcm_rasters\2014_09_2" + os.sep + meter +os.sep+"R01765_" + meter + "_energy.tif"
        corrFile = r"C:\workspace\GLCM\output\new_glcm_rasters\2014_09_2" + os.sep + meter +os.sep+"R01765_" + meter + "_corr.tif"
        ASMFile = r"C:\workspace\GLCM\output\new_glcm_rasters\2014_09_2" + os.sep + meter +os.sep+"R01765_" + meter + "_asm.tif"
        ENTFile = r"C:\workspace\GLCM\output\new_glcm_rasters\2014_09_2" + os.sep + meter +os.sep+"R01765_" + meter + "_entropy.tif"
        meanFile = r"C:\workspace\GLCM\output\new_glcm_rasters\2014_09_2" + os.sep + meter +os.sep+"R01765_" + meter + "_mean.tif"
        varFile = r"C:\workspace\GLCM\output\new_glcm_rasters\2014_09_2" + os.sep + meter +os.sep+"R01765_" + meter + "_var.tif"
        
        #Dont Change anythong below here
        merge, xx, yy, gt = read_raster(in_raster)
        Ny, Nx = np.shape(merge)
        merge[np.isnan(merge)] = 0
        
        Z,ind = sliding_window(merge,(win,win),(win,win))
        
        w = Parallel(n_jobs = 1, verbose=0)(delayed(p_me)(Z[k], win, dist, angle) for k in xrange(len(Z)))
        
        cont = [a[0] for a in w]
        diss = [a[1] for a in w]
        homo = [a[2] for a in w]
        eng  = [a[3] for a in w]
        corr = [a[4] for a in w]
        ASM  = [a[5] for a in w]
        ENT  = [a[6] for a in w]
        mean = [a[7] for a in w]
        var  = [a[8] for a in w]
        del a
        
        #Reshape to match number of windows
        plt_cont = np.reshape(cont , ( ind[0], ind[1] ) )
        plt_diss = np.reshape(diss , ( ind[0], ind[1] ) )
        plt_homo = np.reshape(homo , ( ind[0], ind[1] ) )
        plt_eng = np.reshape(eng , ( ind[0], ind[1] ) )
        plt_corr = np.reshape(corr , ( ind[0], ind[1] ) )
        plt_ASM =  np.reshape(ASM , ( ind[0], ind[1] ) )
        plt_ent = np.reshape(ENT, (ind[0],ind[1]))
        plt_mean =  np.reshape(mean , ( ind[0], ind[1] ) )
        plt_var = np.reshape(var, (ind[0],ind[1]))
        
        
        del cont, diss, homo, eng, corr, ASM, ENT, mean, var
        
#        thing, x,y,gt = read_raster(r"C:\workspace\GLCM\output\glcm_rasters\2014_09_2\3\R01765_3_entropy_resampled.tif")
#        Ny, Nx = np.shape(thing)
#        del x,y
        
        
        #Resize Images to receive texture and define filenames
        contrast = im_resize(plt_cont,Nx,Ny)
        contrast[merge==0]=np.nan
        dissimilarity = im_resize(plt_diss,Nx,Ny)
        dissimilarity[merge==0]=np.nan    
        homogeneity = im_resize(plt_homo,Nx,Ny)
        homogeneity[merge==0]=np.nan
        homogeneity[homogeneity<0]=np.nan
        energy = im_resize(plt_eng,Nx,Ny)
        energy[merge==0]=np.nan
        correlation = im_resize(plt_corr,Nx,Ny)
        correlation[merge==0]=np.nan
        ASM = im_resize(plt_ASM,Nx,Ny)
        ASM[merge==0]=np.nan
        ENT = im_resize(plt_ent,Nx,Ny)
        ENT[merge==0]=np.nan
        ENT[ENT<0]=np.nan
        mean = im_resize(plt_mean,Nx,Ny)
        mean[merge==0]=np.nan
        var = im_resize(plt_var,Nx,Ny)
        var[merge==0]=np.nan
        var[var<0] = np.nan
        del plt_cont, plt_diss, plt_homo, plt_eng, plt_corr, plt_ASM, plt_ent, plt_mean, plt_var
    
        
        del w,Z,ind,Ny,Nx
        
        driverName= 'GTiff'    
        epsg_code=26949
        proj = osr.SpatialReference()
        proj.ImportFromEPSG(epsg_code)
        
        CreateRaster(xx, yy, contrast, gt, proj,driverName,contFile) 
        CreateRaster(xx, yy, dissimilarity, gt, proj,driverName,dissFile)
        CreateRaster(xx, yy, homogeneity, gt, proj,driverName,homoFile)
        CreateRaster(xx, yy, energy, gt, proj,driverName,energyFile)
        CreateRaster(xx, yy, correlation, gt, proj,driverName,corrFile)
        CreateRaster(xx, yy, ASM, gt, proj,driverName,ASMFile)
        CreateRaster(xx, yy, ENT, gt, proj,driverName,ENTFile)
        CreateRaster(xx, yy, mean, gt, proj,driverName,meanFile)
        CreateRaster(xx, yy, var, gt, proj,driverName,varFile)
        
        del contrast, merge, xx, yy, gt, meter, dissimilarity, homogeneity, energy, correlation, ASM, ENT, var, mean