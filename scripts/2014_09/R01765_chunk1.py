# -*- coding: utf-8 -*-
"""
Created on Wed Sep 07 09:36:06 2016

@author: dan
"""

#operational
from __future__ import division
from scipy.io import loadmat 
import os

#numerical
import numpy as np
import PyHum.utils as humutils
from numpy.lib.stride_tricks import as_strided as ast
#from pyhum_utils import sliding_window, im_resize, cut_kmeans
from joblib import Parallel, delayed #, cpu_count
#from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes, grey_erosion
from scipy.interpolate import RectBivariateSpline
import dask.array as da
#import stdev
from skimage.feature import greycomatrix, greycoprops

#plotting
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

    newKernel = RectBivariateSpline(np.r_[:ny],np.r_[:nx],im)
    return newKernel(yy,xx)
    
def p_me(Z):
    '''
    loop to calc glcm stuff
    '''
    try:
        glcm = greycomatrix(Z, [5], [0], 256, symmetric=True, normed=True)
        cont = greycoprops(glcm, 'contrast')
        diss = greycoprops(glcm, 'dissimilarity')
        homo = greycoprops(glcm, 'homogeneity')
        eng = greycoprops(glcm, 'energy')
        corr = greycoprops(glcm, 'correlation')
        ASM = greycoprops(glcm, 'ASM')
        return (cont, diss, homo, eng, corr, ASM)
    except:
        print 'Something went wrong'
        return (0,0,0,0,0,0)
        


def get_mmap_data(sonpath, base, string, dtype, shape):
    #we are only going to access the portion of memory required
    with open(os.path.normpath(os.path.join(sonpath,base+string)), 'r') as ff:
       fp = np.memmap(ff, dtype=dtype, mode='r', shape=shape)
    return fp      

def set_mmap_data(sonpath, base, string, dtype, Zt):
    # create memory mapped file for Z
    #with open(os.path.normpath(os.path.join(sonpath,base+string)), 'w+') as ff:
    #   fp = np.memmap(ff, dtype=dtype, mode='w+', shape=np.shape(Zt))
    try:
       os.remove(os.path.normpath(os.path.join(sonpath,base+string)))
    except:
       pass

    try:
       with open(os.path.normpath(os.path.join(sonpath,base+string)), 'w+') as ff:
          fp = np.memmap(ff, dtype=dtype, mode='readwrite', shape=np.shape(Zt))
       fp[:] = Zt[:]

    except:
       with open(os.path.normpath(os.path.join(sonpath,base+string)), 'w+') as ff:
          fp = np.memmap(ff, dtype=dtype, mode='copyonwrite', shape=np.shape(Zt))
       fp[:] = Zt[:]

    del fp
    shape = np.shape(Zt)
    del Zt
    return shape  
    
    
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



if __name__ == '__main__':
    humfile = r"C:\workspace\Reach_4a\2014_09\analysisJan2016\R01765\R01765.DAT"
    sonpath = r"C:\workspace\Reach_4a\2014_09\analysisJan2016\R01765"
    
    win = 31 
    
    base = humfile.split('.DAT') # get base of file name for output
    base = base[0].split(os.sep)[-1]
    
    base = humutils.strip_base(base)
    
    meta = loadmat(os.path.normpath(os.path.join(sonpath,base+'meta.mat')))
    
    # load memory mapped scans
    shape_port = np.squeeze(meta['shape_port'])
    if shape_port!='':
       #port_fp = np.memmap(sonpath+base+'_data_port_la.dat', dtype='float32', mode='r', shape=tuple(shape_port))
       with open(os.path.normpath(os.path.join(sonpath,base+'_data_port_lar.dat')), 'r') as ff:
          port_fp = np.memmap(ff, dtype='float32', mode='r', shape=tuple(shape_port))
    
    shape_star = np.squeeze(meta['shape_star'])
    if shape_star!='':
       #star_fp = np.memmap(sonpath+base+'_data_star_la.dat', dtype='float32', mode='r', shape=tuple(shape_star))
       with open(os.path.normpath(os.path.join(sonpath,base+'_data_star_lar.dat')), 'r') as ff:
          star_fp = np.memmap(ff, dtype='float32', mode='r', shape=tuple(shape_star))
    
    dist_m = np.squeeze(meta['dist_m'])
    ft = 1/(meta['pix_m'])
    extent = shape_star[1] 
    
    #Start of GLCM stuff
    
    Zs = []; Zp = []
    
    #if len(np.shape(star_fp))>2:
    #     for p in xrange(len(star_fp)):
    p=1
     
    #Creates float 32 array
    merge = np.vstack((np.flipud(port_fp[p]),star_fp[p]))
     
    #change to float 64 array
    merge = np.asarray(merge, 'float64')
    
    merge_mask = np.vstack((np.flipud(port_fp[p]),star_fp[p]))
    
    #why are we masking it with          
    merge[merge_mask==0] = 0
    del merge_mask
    
    #change to 8 bit precision
    mask = np.asarray(merge!=0,'int8') # only 8bit precision needed
    
    merge[np.isnan(merge)] = 0
    
    Z,ind = sliding_window(merge,(win,win),(win,win))
    
    Ny, Nx = np.shape(merge)
    
    
    w = Parallel(n_jobs = 1, verbose=0)(delayed(p_me)(Z[k]) for k in xrange(len(Z)))
     
    cont = [a[0] for a in w]
    diss = [a[1] for a in w]
    homo = [a[2] for a in w]
    eng  = [a[3] for a in w]
    corr = [a[4] for a in w]
    ASM  = [a[5] for a in w]
    
    del w
    
    #Reshape to match number of windows
    plt_cont = np.reshape(cont , ( ind[0], ind[1] ) )
    plt_diss = np.reshape(diss , ( ind[0], ind[1] ) )
    plt_homo = np.reshape(homo , ( ind[0], ind[1] ) )
    plt_eng = np.reshape(eng , ( ind[0], ind[1] ) )
    plt_corr = np.reshape(corr , ( ind[0], ind[1] ) )
    plt_ASM =  np.reshape(ASM , ( ind[0], ind[1] ) )
    del cont, diss, homo, eng, corr, ASM
    
    #Resize Images to receive texture and define filenames
    contrast = im_resize(plt_cont,Nx,Ny)
    contrast[merge==0]=np.nan
    dissimilarity = im_resize(plt_diss,Nx,Ny)
    dissimilarity[merge==0]=np.nan    
    homogeneity = im_resize(plt_homo,Nx,Ny)
    homogeneity[merge==0]=np.nan
    energy = im_resize(plt_eng,Nx,Ny)
    energy[merge==0]=np.nan
    correlation = im_resize(plt_corr,Nx,Ny)
    correlation[merge==0]=np.nan
    ASM = im_resize(plt_ASM,Nx,Ny)
    ASM[merge==0]=np.nan
    del plt_cont, plt_diss, plt_homo, plt_eng, plt_corr, plt_ASM
    
    img_plot = merge
    img_plot[merge==0]=np.nan
    del merge

    # Create figure to receive results
    fig = plt.figure(figsize=(18,5))
    fig.suptitle('GLCM Textures')
    
    #Plot input image
    Zdist = dist_m
    
    ax5 = plt.subplot(245)
    ax6 = plt.subplot(246, sharey=ax5)
    ax7 = plt.subplot(247, sharey=ax5)
    ax = plt.subplot(241, sharex=ax5)
    ax2 = plt.subplot(242, sharey=ax,sharex=ax6)
    ax3 = plt.subplot(243, sharey=ax, sharex =ax7)
    ax4 = plt.subplot(244, sharey=ax)

    ax.set_title('Original Image')    
    img = ax.imshow(img_plot, cmap='gray', extent=[min(Zdist), max(Zdist), -extent*(1/ft), extent*(1/ft)])  
    ax.set_ylabel('Range (m)')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbr = plt.colorbar(img,cax=cax)
    ax.set_adjustable('box-forced')

    ax2.set_title('Contrast')
    img2 = ax2.imshow(contrast, cmap = 'gray', extent=[min(Zdist), max(Zdist), -extent*(1/ft), extent*(1/ft)])
    ax2.set_adjustable('box-forced')        
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbr = plt.colorbar(img2,cax=cax)
    
    ax3.set_title('Dissimilarity')
    img3 = ax3.imshow(dissimilarity, cmap = 'gray', extent=[min(Zdist), max(Zdist), -extent*(1/ft), extent*(1/ft)])
    ax3.set_adjustable('box-forced')    
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.1) 
    cbr = plt.colorbar(img3,cax=cax)
    
    ax4.set_title('Homogeneity')
    im4 = ax4.imshow(homogeneity, cmap = 'gray', extent=[min(Zdist), max(Zdist), -extent*(1/ft), extent*(1/ft)])
    ax4.set_xbound(lower = min(Zdist), upper=max(Zdist))
    ax4.set_xlabel('Distance along track (m)')
    ax4.set_adjustable('box-forced')
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", size="5%", pad=0.1)  
    cbr = plt.colorbar(im4,cax=cax)
    
    ax5.set_title('Energy')
    im5 = ax5.imshow(energy, cmap = 'gray', extent=[min(Zdist), max(Zdist), -extent*(1/ft), extent*(1/ft)])
    ax5.set_ylabel('Range (m)')
    ax5.set_xlabel('Distance along track (m)')
    ax5.set_adjustable('box-forced')
    divider = make_axes_locatable(ax5)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbr = plt.colorbar(im5,cax=cax)
    
    ax6.set_title('Correlation')
    im6 = ax6.imshow(correlation, cmap = 'gray', extent=[min(Zdist), max(Zdist), -extent*(1/ft), extent*(1/ft)])    
    ax6.set_xlabel('Distance along track (m)')
    ax6.set_adjustable('box-forced')    
    divider = make_axes_locatable(ax6)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbr = plt.colorbar(im6,cax=cax)
    
    ax7.set_title('ASM')
    im7 = ax7.imshow(ASM, cmap = 'gray', extent=[min(Zdist), max(Zdist), -extent*(1/ft), extent*(1/ft)])
    ax7.set_xlabel('Distance along track (m)')
    ax7.set_adjustable('box-forced')    
    divider = make_axes_locatable(ax7)
    cax = divider.append_axes("right", size="5%", pad=0.1) 
    cbr = plt.colorbar(im7,cax=cax)
    
    plt.tight_layout()
    #plt.show()
    
    plt.savefig(r"C:\workspace\GLCM\output\2014_09\R01765_chunk1.png",dpi=1000)
         
         
         
         

