# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 14:29:23 2016

@author: dan
"""
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
import pyproj
from rasterstats import zonal_stats
import pandas as pd
import ogr
import os
import gdal
import numpy as np
from pandas.tools.plotting import table


def make_df(x):
    df = pd.DataFrame(x.compressed())
    return df
def make_df2(x):
    df = pd.DataFrame(x,columns=['dBW'])
    return df
def mykurt(x):
    df = make_df(x)
    return float(df.kurtosis().values)
    
def myskew(x):
    df = make_df(x)
    skew = df.skew().values
    return float(skew[0])
    
def sort_hack(row):
    if row['substrate']=='sand':
        return 1
    if row['substrate']=='sand/gravel':
        return 2
    if row['substrate']=='gravel':
        return 3
    if row['substrate']=='gravel/sand':
        return 4
    if row['substrate']=='gravel/boulders':
        return 5
    if row['substrate']=='boulders':
        return 6   



#things that wont change
win_sizes = [8,12,20,40]
in_shp_09 = r"C:\workspace\Merged_SS\window_analysis\shapefiles\tex_seg_2014_09_67_3class.shp"
in_shp_04 = r"C:\workspace\Merged_SS\window_analysis\shapefiles\tex_seg_800.shp"
geo_shp_09 = r"C:\workspace\Merged_SS\window_analysis\shapefiles\tex_seg_2014_09_67_geo"
geo_shp_04 = r"C:\workspace\Merged_SS\window_analysis\shapefiles\tex_seg_800_geo"
root_04 = r"C:\workspace\GLCM\output\glcm_rasters\2014_04"
root_09 = r"C:\workspace\GLCM\output\glcm_rasters\2014_09"
root_plot = r"C:\workspace\GLCM\output\glcm_plots"



#Plotting Variables
a_val = 0.6
colors = ['#ef8a62','#f7f7f7','#67a9cf']
circ1 = Line2D([0], [0], linestyle="none", marker="o", markersize=10, markerfacecolor=colors[0],alpha=a_val)
circ2 = Line2D([0], [0], linestyle="none", marker="o", markersize=10, markerfacecolor=colors[1],alpha=a_val)
circ3 = Line2D([0], [0], linestyle="none", marker="o", markersize=10, markerfacecolor=colors[2],alpha=a_val)
cs2cs_args = "epsg:26949"

trans =  pyproj.Proj(init="epsg:26949") 
wms_url = r"http://grandcanyon.usgs.gov/arcgis/services/Imagery/ColoradoRiverImageryExplorer/MapServer/WmsServer?"
font_size=10

#change this stuff
suffix = "_diss.tif"
cbr_txt = "Dissimilarity"
c_ramp = 'hot'
for win_size in win_sizes[1:]:
    print 'Now working in %s window size' %(win_size,)
    
    #calulate window size in meters
    meter = win_size/4    
    #build file paths to input rasters
    glcm_09 = root_09 + os.sep + str(meter) + os.sep+ "R01767_"+ str(meter) + suffix
    glcm_04 = root_04 + os.sep + str(meter) + os.sep+ "R01346_R01347_"+ str(meter) + suffix

    #########################################################################################################################
    ########################################################################################################################
    #                   2014_09 zonal stats
    #########################################################################################################################
    #########################################################################################################################
    #Calculate zonal statistics for the spet visual segmentaion
    z_stats_09 = zonal_stats(in_shp_09 ,glcm_09,stats=['count'],raster_out=True)

    #Lets get get the substrate to sort lists
    ds = ogr.Open(in_shp_09)
    lyr = ds.GetLayer(0)
    a=[]
    for row in lyr:
        a.append(row.substrate)
    lyr.ResetReading()
    del ds

    s, g, b = [],[],[]
    n = 0
    for item in z_stats_09:
        raster_array = item['mini_raster_array'].compressed()
        substrate = a[n]
        if substrate=='sand':
            s.extend(list(raster_array))
        if substrate=='gravel':
            g.extend(list(raster_array))
        if substrate=='boulders':
            b.extend(list(raster_array))
        n+=1
    del raster_array, substrate, n, item, a

    #########################################################################################################################
    ########################################################################################################################
    #                   2014_04 zonal stats
    #########################################################################################################################
    #########################################################################################################################
    
    #Calculate zonal statistics for the spet visual segmentaion
    z_stats_04 = zonal_stats(in_shp_04 ,glcm_04,stats=['count'],raster_out=True)

    #Lets get get the substrate to sort lists
    ds = ogr.Open(in_shp_04)
    lyr = ds.GetLayer(0)
    a=[]
    for row in lyr:
        a.append(row.substrate)
    lyr.ResetReading()
    del ds

    n = 0
    for item in z_stats_09:
        raster_array = item['mini_raster_array'].compressed()
        substrate = a[n]
        if substrate=='sand':
            s.extend(list(raster_array))
        if substrate=='gravel':
            g.extend(list(raster_array))
        if substrate=='boulders':
            b.extend(list(raster_array))
        n+=1
    del raster_array, substrate, n, item, a
    
    s_df = make_df2(s)
    g_df = make_df2(g)
    b_df = make_df2(b)
    del s, g, b, z_stats_04, z_stats_09
    
    #########################################################################################################################
    ########################################################################################################################
    #                   Aggregrated zonal statistics
    #########################################################################################################################
    #########################################################################################################################

    tbl = pd.DataFrame(columns=['substrate','10%','20%','25%','50%','75%','kurt','skew'])
    tbl['substrate']=['sand','gravel','boulders']
    tbl = tbl.set_index('substrate')
    tbl.loc['sand'] = pd.Series({'10%':float(s_df.quantile(0.1).values),'20%':float(s_df.quantile(0.2).values),'25%':float(s_df.describe().iloc[4].values), '50%':float(s_df.describe().iloc[5].values),'75%':float(s_df.describe().iloc[6].values),'kurt':float(s_df.kurtosis().values),'skew':float(s_df.skew().values)})
    tbl.loc['gravel'] = pd.Series({'10%':float(g_df.quantile(0.1).values),'20%':float(g_df.quantile(0.2).values),'25%':float(g_df.describe().iloc[4].values), '50%':float(g_df.describe().iloc[5].values),'75%':float(g_df.describe().iloc[6].values),'kurt':float(g_df.kurtosis().values),'skew':float(g_df.skew().values)})
    tbl.loc['boulders'] = pd.Series({'10%':float(b_df.quantile(0.1).values),'20%':float(b_df.quantile(0.2).values),'25%':float(b_df.describe().iloc[4].values), '50%':float(b_df.describe().iloc[5].values),'75%':float(b_df.describe().iloc[6].values),'kurt':float(b_df.kurtosis().values),'skew':float(b_df.skew().values)})
    tbl = tbl.applymap(lambda x: round(x,3))
    del s_df, g_df, b_df
    
    #########################################################################################################################
    ########################################################################################################################
    #                   Begin Plotting Routine
    #########################################################################################################################
    #########################################################################################################################
    
    #Get plotting extents for 2014_09
    ds = gdal.Open(glcm_09)
    data_09 = ds.GetRasterBand(1).ReadAsArray()
    data_09[data_09<=0] = np.nan
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    xres = gt[1]
    yres = gt[5]
    xmin = gt[0] + xres * 0.5
    xmax = gt[0] + (xres * ds.RasterXSize) - xres * 0.5
    ymin = gt[3] + (yres * ds.RasterYSize) + yres * 0.5
    ymax = gt[3] - yres * 0.5
    extent = [xmin,xmax,ymin,ymax]
    del ds
    
    xx, yy = np.mgrid[xmin:xmax+xres:xres, ymax+yres:ymin:yres]
    glon_09, glat_09 = trans(xx, yy, inverse=True)
    del xx, yy, xres, yres, gt, xmin, xmax, ymin, ymax
    
    #Get plotting extents for 2014_04
    ds = gdal.Open(glcm_04)
    data_04 = ds.GetRasterBand(1).ReadAsArray()
    data_04[data_04<=0] = np.nan
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    xres = gt[1]
    yres = gt[5]
    xmin = gt[0] + xres * 0.5
    xmax = gt[0] + (xres * ds.RasterXSize) - xres * 0.5
    ymin = gt[3] + (yres * ds.RasterYSize) + yres * 0.5
    ymax = gt[3] - yres * 0.5
    extent = [xmin,xmax,ymin,ymax]
    del ds
    
    xx, yy = np.mgrid[xmin:xmax+xres:xres, ymax+yres:ymin:yres]
    glon_04, glat_04 = trans(xx, yy, inverse=True)
    del xx, yy, xres, yres, gt, xmin, xmax, ymin, ymax 
    
    title_09 = "September 2014: R01767 " + "\n"+ str(meter) + " meter grid"
    title_04 = "April 2014: R01346 R01347 " + "\n" + str(meter) + " meter grid"
    
    
    #########################################################################################################################
    ########################################################################################################################
    #                   Begin Subplot 1
    #########################################################################################################################
    #########################################################################################################################
    print 'Now Plotting April 2014...'
    fig = plt.figure(figsize=(9,10))
    ax = plt.subplot2grid((10,2),(0, 0),rowspan=9)
    ax.set_title(title_04)
    m = Basemap(projection='merc', 
                epsg=cs2cs_args.split(':')[1], 
                llcrnrlon=np.min(glon_04) - 0.0004, 
                llcrnrlat=np.min(glat_04) - 0.0006,
                urcrnrlon=np.max(glon_04) + 0.0006, 
                urcrnrlat=np.max(glat_04) + 0.0009)
    x,y = m.projtran(glon_04, glat_04)
    m.wmsimage(server=wms_url, layers=['3'], xpixels=1000)
    im = m.contourf(x,y,data_04.T, cmap=c_ramp)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbr = plt.colorbar(im, cax=cax)
    cbr.set_label(cbr_txt, size=10)
    #read shapefile and create polygon collections
    m.readshapefile(geo_shp_04,"layer",drawbounds = False)
    
    #sand, gravel, boulders
    s_patch, g_patch, b_patch  =[],[],[]
    
    for info, shape in zip(m.layer_info, m.layer):
        if info['substrate'] == 'sand':
            s_patch.append(Polygon(np.asarray(shape),True))   
        if info['substrate'] == 'gravel':
            g_patch.append(Polygon(np.asarray(shape),True))               
        if info['substrate'] == 'boulders':
            b_patch.append(Polygon(np.asarray(shape),True))
    del info, shape
    
    ax.add_collection(PatchCollection(s_patch, facecolor = colors[0],alpha=a_val, edgecolor='none',zorder=10))
    ax.add_collection(PatchCollection(g_patch, facecolor = colors[1],alpha=a_val, edgecolor='none',zorder=10)) 
    ax.add_collection(PatchCollection(b_patch, facecolor = colors[2],alpha=a_val, edgecolor='none',zorder=10))
    del s_patch, g_patch, b_patch
    
    ax.legend((circ1, circ2, circ3), ("sand", "gravel", "boulders"), numpoints=1, loc='best', borderaxespad=0., fontsize=font_size) 
    
    #########################################################################################################################
    ########################################################################################################################
    #                   Begin Subplot 2
    #########################################################################################################################
    #########################################################################################################################
    print 'Now Plotting September 2014...'
    ax = plt.subplot2grid((10,2),(0, 1),rowspan=9)
    ax.set_title(title_09)
    m = Basemap(projection='merc', 
                epsg=cs2cs_args.split(':')[1], 
                llcrnrlon=np.min(glon_09) - 0.0002, 
                llcrnrlat=np.min(glat_09) - 0.0006,
                urcrnrlon=np.max(glon_09) + 0.0002, 
                urcrnrlat=np.max(glat_09) + 0.0006)
    x,y = m.projtran(glon_09, glat_09)
    m.wmsimage(server=wms_url, layers=['3'], xpixels=1000)
    im = m.contourf(x,y,data_09.T, cmap=c_ramp)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbr = plt.colorbar(im, cax=cax)
    cbr.set_label(cbr_txt, size=10)
    #read shapefile and create polygon collections
    m.readshapefile(geo_shp_09,"layer",drawbounds = False)
    
    #sand, gravel, boulders
    s_patch, g_patch, b_patch  =[],[],[]
    
    for info, shape in zip(m.layer_info, m.layer):
        if info['substrate'] == 'sand':
            s_patch.append(Polygon(np.asarray(shape),True))   
        if info['substrate'] == 'gravel':
            g_patch.append(Polygon(np.asarray(shape),True))               
        if info['substrate'] == 'boulders':
            b_patch.append(Polygon(np.asarray(shape),True))
    del info, shape
    
    ax.add_collection(PatchCollection(s_patch, facecolor = colors[0],alpha=a_val, edgecolor='none',zorder=10))
    ax.add_collection(PatchCollection(g_patch, facecolor = colors[1],alpha=a_val, edgecolor='none',zorder=10)) 
    ax.add_collection(PatchCollection(b_patch, facecolor = colors[2],alpha=a_val, edgecolor='none',zorder=10))
    del s_patch, g_patch, b_patch
    
    ax.legend((circ1, circ2, circ3), ("sand", "gravel", "boulders"), numpoints=1, loc='best', borderaxespad=0., fontsize=font_size) 
    
    #########################################################################################################################
    ########################################################################################################################
    #                   Begin Subplot 3: Table
    #########################################################################################################################
    #########################################################################################################################
    
    ax2 = plt.subplot2grid((10,2),(9, 0), colspan=2)
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    for sp in ax2.spines.itervalues():
        sp.set_color('w')
        sp.set_zorder(0)
    the_table = table(ax2, tbl,loc='upper right',colWidths=[0.1,0.1,0.1,0.1,0.1,0.1,0.1])
    the_table.set_fontsize(10)
    
    print 'Now Saving Figure...'
    out_fig = root_plot + os.sep + str(meter) + os.sep + cbr_txt + "_" + str(meter)+".png"
    plt.suptitle(cbr_txt)
    plt.savefig(out_fig, dpi=600)


    
    