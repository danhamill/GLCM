# -*- coding: utf-8 -*-
"""
Created on Tue Nov 08 11:16:33 2016

@author: dan
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from skimage import img_as_float
from skimage import exposure
import gdal
import pandas as pd
matplotlib.rcParams['font.size'] = 8


def plot_img_and_hist(img, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    img = img_as_float(img)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(img, cmap=plt.cm.gray)
    ax_img.set_axis_off()
    ax_img.set_adjustable('box-forced')

    # Display histogram
    ax_hist.hist(img.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(img, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf

 

in_raster = r"C:\workspace\Merged_SS\raster\2014_09\ss_2014_09_R01767_raster.tif"

ds =gdal.Open(in_raster)
img = ds.GetRasterBand(1).ReadAsArray()
del ds

fig = plt.figure(figsize=(10,5))
axes = np.zeros((2,4), dtype=np.object)
axes[0,0] = fig.add_subplot(2, 4, 1)
for i in range(1,4):
    axes[0,i] = fig.add_subplot(2, 4, 1+i,  sharex=axes[0,0], sharey=axes[0,0])
for i in range(0,4):
    axes[1,i] = fig.add_subplot(2, 4, 5+i)

img[img == -99] = 0
raw_plot = np.copy(img)
raw_plot[img==0]=np.nan
axes[0,0].imshow(raw_plot,cmap=plt.cm.gray_r)
axes[0,0].set_axis_off()
axes[0,0].set_adjustable('box-forced')
pd.DataFrame({'data':img[img!=0].flatten()}).plot.hist(bins=50, title='Original Image',ax=axes[1,0],xlim=(0,30),legend=False,histtype='step')
ax_cdf = axes[1,0].twinx()
ax_cdf.set_ylabel('')
pd.DataFrame({'data':img[img!=0].flatten()}).plot.hist(bins=50, title='Original Image',ax=ax_cdf,xlim=(0,30),legend=False,histtype='step',cumulative='True',color='r')
ax_cdf.set_yticks([])
ax_cdf.set_ylabel('')


#convert to int8 precision
img = np.asarray(img,'int8')
#img_plot = np.copy(img).astype('float32')
#img_plot[np.isnan(raw_plot)]=np.nan
#axes[0,1].imshow(img_plot,cmap=plt.cm.gray_r)
#axes[0,1].set_axis_off()
#axes[0,1].set_adjustable('box-forced')
#pd.DataFrame({'data':img[img!=0].flatten()}).plot.hist(bins=50, title='INT8 image',ax=axes[1,1],xlim=(0,35),legend=False,histtype='step')
#ax_cdf = axes[1,0].twinx()
#pd.DataFrame({'data':img[img!=0].flatten()}).plot.hist(bins=50, title='Original Image',ax=ax_cdf,xlim=(0,30),legend=False,histtype='step',cumulative='True',color='r')
#del img_plot

#Rescale Image
p2, p98 = np.percentile(img[img!=0], (2, 98))
img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
img_plot = np.copy(img_rescale).astype('float32')
img_plot[np.isnan(raw_plot)]=np.nan
axes[0,1].imshow(img_plot,cmap=plt.cm.gray_r)
axes[0,1].set_axis_off()
axes[0,1].set_adjustable('box-forced')
pd.DataFrame({'data':img_rescale[img!=0].flatten()}).plot.hist(bins=50, title='Rescaled Image',ax=axes[1,1],legend=False,histtype='step')
ax_cdf = axes[1,1].twinx()
pd.DataFrame({'data':img_rescale[img!=0].flatten()}).plot.hist(bins=50, title='Rescaled Image',ax=ax_cdf,legend=False,histtype='step',cumulative='True',color='r')
ax_cdf.set_ylabel('')
ax_cdf.set_yticks([])
del img_plot

# Equalization
img_eq = exposure.equalize_hist(img)
img_plot = np.copy(img_eq).astype('float32')
img_plot[np.isnan(raw_plot)]=np.nan
axes[0,2].imshow(img_plot,cmap=plt.cm.gray_r)
axes[0,2].set_axis_off()
axes[0,2].set_adjustable('box-forced')
pd.DataFrame({'data':img_eq[img!=0].flatten()}).plot.hist(bins=50, title='Equalized Image',ax=axes[1,2],xlim=(0.82,0.999),legend=False,histtype='step')
ax_cdf = axes[1,2].twinx()
pd.DataFrame({'data':img_eq[img!=0].flatten()}).plot.hist(bins=50, title='Equalized Image',ax=ax_cdf,xlim=(0.82,0.999),legend=False,histtype='step',cumulative='True',color='r')
ax_cdf.set_ylabel('')
ax_cdf.set_yticks([])
del img_plot

# Adaptive Equalization
img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.04)
img_plot = np.copy(img_adapteq).astype('float32')
img_plot[np.isnan(raw_plot)]=np.nan
axes[0,3].imshow(img_plot,cmap=plt.cm.gray_r)
axes[0,3].set_axis_off()
axes[0,3].set_adjustable('box-forced')
pd.DataFrame({'data':img_adapteq[img!=0].flatten()}).plot.hist(bins=50, title='Apaptive Equalized Image',ax=axes[1,3],legend=False,histtype='step')
ax_cdf = axes[1,3].twinx()
pd.DataFrame({'data':img_adapteq[img!=0].flatten()}).plot.hist(bins=50, title='Apaptive Equalized Image',ax=ax_cdf,legend=False,histtype='step',cumulative='True',color='r')
ax_cdf.set_ylabel('')
ax_cdf.set_yticks([])
del img_plot

fig.subplots_adjust(wspace=0.4)
plt.tight_layout()
plt.show()



###################################################################################################################################################################

# Display results
fig = plt.figure(figsize=(8, 5))
axes = np.zeros((2,4), dtype=np.object)
axes[0,0] = fig.add_subplot(2, 4, 1)
for i in range(1,4):
    axes[0,i] = fig.add_subplot(2, 4, 1+i, sharex=axes[0,0], sharey=axes[0,0])
for i in range(0,4):
    axes[1,i] = fig.add_subplot(2, 4, 5+i)

    
    
    
ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, 0])
ax_img.set_title('Low contrast image')

y_min, y_max = ax_hist.get_ylim()
ax_hist.set_ylabel('Number of pixels')
ax_hist.set_yticks(np.linspace(0, y_max, 5))

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale[img!=0], axes[:, 1])
ax_img.set_title('Contrast stretching')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq[img!=0], axes[:, 2])
ax_img.set_title('Histogram equalization')

ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_adapteq[img!=0], axes[:, 3])
ax_img.set_title('Adaptive equalization')

ax_cdf.set_ylabel('Fraction of total intensity')
ax_cdf.set_yticks(np.linspace(0, 1, 5))

# prevent overlap of y-axis labels
fig.subplots_adjust(wspace=0.4)
plt.show()