# -*- coding: utf-8 -*-
"""
Created on Tue May 02 12:11:06 2017

@author: dan
"""

from osgeo import gdal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import itemfreq
import pyproj
from mpl_toolkits.basemap import Basemap
from matplotlib.lines import Line2D

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

def count_unique(keys):
    uniq_keys = np.unique(keys)
    bins = uniq_keys.searchsorted(keys)
    return uniq_keys, np.bincount(bins)

def get_sedclass_counts(df):
    try:
        s = df[df['sedclass']==1]['counts'].iloc[0]
    except:
        s=0
    try:
        g = df[df['sedclass']==2]['counts'].iloc[0]
    except:
        g=0    
    try:
        b = df[df['sedclass']==3]['counts'].iloc[0]
    except:
        b=0
    return s,g,b


gmm2_raster = r"C:\workspace\GLCM\Neigiborhood_Analysis\Sedclass_Rasters\R02028_GMM_2class_raster.tif"
gmm3_raster = r"C:\workspace\GLCM\Neigiborhood_Analysis\Sedclass_Rasters\R02028_GMM_3class_raster.tif"
lsq_raster = r"C:\workspace\GLCM\Neigiborhood_Analysis\Sedclass_Rasters\R02028_median_Sed_Class_3_variable.tif"


gmm2_data,xx,yy, gt = read_raster(gmm2_raster)
gmm3_data = read_raster(gmm3_raster)[0]
lsq_data = read_raster(lsq_raster)[0]

df = pd.DataFrame(columns=['Pixels_Down_Stream','Sand','Gravel','Boulders','Model'])

for i in list(range(np.shape(gmm2_data)[0])):
    gmm2_tmp = gmm2_data.copy()
    gmm3_tmp = gmm3_data.copy()
    lsq_tmp = lsq_data.copy()
    
    gmm2_counts = itemfreq(gmm2_tmp[0:i+1,:])[~np.isnan(itemfreq(gmm2_tmp[0:i+1,:])).any(axis=1)]
    gmm3_counts = itemfreq(gmm3_tmp[0:i+1,:])[~np.isnan( itemfreq(gmm3_tmp[0:i+1,:])).any(axis=1)]
    lsq_counts =  itemfreq(lsq_tmp[0:i+1,:])[~np.isnan( itemfreq(lsq_tmp[0:i+1,:])).any(axis=1)]
    
    lsq_df = pd.DataFrame(lsq_counts,columns=['sedclass','counts'])
    gmm2_df = pd.DataFrame(gmm2_counts,columns=['sedclass','counts'])
    gmm3_df = pd.DataFrame(gmm3_counts,columns=['sedclass','counts'])
    
    s,g,b = get_sedclass_counts(lsq_df)
    
    distance = (i+1)*3
    
    df =df.append(pd.Series({'Pixels_Down_Stream':distance, 'Sand':s, 'Gravel':g,'Boulders':b,'Model':'LSQ'}),ignore_index=True)
    s,g,b = get_sedclass_counts(gmm2_df)
    df =df.append(pd.Series({'Pixels_Down_Stream':distance, 'Sand':s, 'Gravel':g,'Boulders':b,'Model':'gmm2'}),ignore_index=True)
    s,g,b = get_sedclass_counts(gmm3_df)
    df =df.append(pd.Series({'Pixels_Down_Stream':distance, 'Sand':s, 'Gravel':g,'Boulders':b,'Model':'gmm3'}),ignore_index=True)
    
    
df['sum'] = df.loc[:,'Sand':'Boulders'].sum(axis=1)
df_new = df.loc[:,'Sand':'Boulders'].div(df["sum"], axis=0)
df_new['Model'] = df['Model']
df_new['Distance'] = df['Pixels_Down_Stream']
df_new = df_new.dropna()

fig,(ax,ax1,ax2) = plt.subplots(nrows=3,sharey=True)
df_new[df_new.Model == 'LSQ'].plot.line(ax=ax, x='Distance',ylim=(0,1),sharex=ax2)
df_new[df_new.Model == 'gmm2'].plot.line(ax=ax1, x='Distance',ylim=(0,1),sharex=ax2)
df_new[df_new.Model == 'gmm3'].plot.line(ax=ax2, x='Distance',ylim=(0,1))
ax2.set_xlabel('Distance Down Stream (m)')
ax.legend(loc=9,ncol=3,fontsize=10)
ax1.legend(loc=9,ncol=3,fontsize=10)
ax2.legend(loc=9,ncol=3,fontsize=10)

ax.text(1732,0.8, 'A',fontsize=18,color='black',backgroundcolor='w')
ax1.text(1732,0.8, 'B',fontsize=18,color='black',backgroundcolor='w')
ax2.text(1732,0.8, 'C',fontsize=18,color='black',backgroundcolor='w')

plt.tight_layout()
plt.savefig(r"C:\workspace\GLCM\Neigiborhood_Analysis\Output\dist_dwn_stream.png",dpi=600)

trans =  pyproj.Proj(init="epsg:26949") 
glon, glat = trans(xx, yy, inverse=True)
#ortho_lon, ortho_lat = trans(ortho_x, ortho_y, inverse=True)
cs2cs_args = "epsg:26949"
a_val = 1
colors = ['#EA5739','#FEFFBE','#4BB05C']
circ1 = Line2D([0], [0], linestyle="none", marker="s", markersize=10, markerfacecolor=colors[0],alpha=a_val)
circ3 = Line2D([0], [0], linestyle="none", marker="s", markersize=10, markerfacecolor=colors[1],alpha=a_val)
circ4 = Line2D([0], [0], linestyle="none", marker="s", markersize=10, markerfacecolor=colors[2],alpha=a_val)


fig,(ax,ax1,ax2)= plt.subplots(ncols=3)

m = Basemap(projection='merc', 
epsg=cs2cs_args.split(':')[1], 
llcrnrlon=np.min(glon-0.0004), 
llcrnrlat=np.min(glat-0.0004),
urcrnrlon=np.max(glon+0.0008), 
urcrnrlat=np.max(glat+0.0004),ax=ax)
m.wmsimage(server='http://grandcanyon.usgs.gov/arcgis/services/Imagery/ColoradoRiverImageryExplorer/MapServer/WmsServer?', layers=['3'], xpixels=1000)
x,y = m.projtran(glon, glat)
m.contourf(x,y,lsq_data.T, cmap='RdYlGn',levels=[0,1,2,3])
m.drawmapscale(np.min(glon)+0.001, np.min(glat)+0.0030, np.min(glon), np.min(glat), 200., units='m', barstyle='fancy', labelstyle='simple', fontcolor='black')
ax.legend((circ1, circ3,circ4), ("1 = sand", "2 = gravel","3 = boulders"), numpoints=1, loc=1, borderaxespad=0., fontsize=8)  

m = Basemap(projection='merc', 
epsg=cs2cs_args.split(':')[1], 
llcrnrlon=np.min(glon-0.0004), 
llcrnrlat=np.min(glat-0.0004),
urcrnrlon=np.max(glon+0.0008), 
urcrnrlat=np.max(glat+0.0004),ax=ax1)
m.wmsimage(server='http://grandcanyon.usgs.gov/arcgis/services/Imagery/ColoradoRiverImageryExplorer/MapServer/WmsServer?', layers=['3'], xpixels=1000)
x,y = m.projtran(glon, glat)
m.contourf(x,y,gmm2_data.T, cmap='RdYlGn',levels=[0,1,2,3])
m.drawmapscale(np.min(glon)+0.001, np.min(glat)+0.0030, np.min(glon), np.min(glat), 200., units='m', barstyle='fancy', labelstyle='simple', fontcolor='black')
ax1.legend((circ1, circ4), ("1 = sand", "3 = boulders"), numpoints=1, loc=1, borderaxespad=0., fontsize=8)  

m = Basemap(projection='merc', 
epsg=cs2cs_args.split(':')[1], 
llcrnrlon=np.min(glon-0.0004), 
llcrnrlat=np.min(glat-0.0004),
urcrnrlon=np.max(glon+0.0008), 
urcrnrlat=np.max(glat+0.0004),ax=ax2)
m.wmsimage(server='http://grandcanyon.usgs.gov/arcgis/services/Imagery/ColoradoRiverImageryExplorer/MapServer/WmsServer?', layers=['3'], xpixels=1000)
x,y = m.projtran(glon, glat)
m.contourf(x,y,gmm3_data.T, cmap='RdYlGn',levels=[0,1,2,3])
m.drawmapscale(np.min(glon)+0.001, np.min(glat)+0.0030, np.min(glon), np.min(glat), 200., units='m', barstyle='fancy', labelstyle='simple', fontcolor='black')
ax2.legend((circ1, circ3,circ4), ("1 = sand", "2 = gravel","3 = boulders"), numpoints=1, loc=1, borderaxespad=0., fontsize=8)  

ax.text(78,603, 'A',fontsize=18,color='black',backgroundcolor='w')
ax1.text(78,603, 'B',fontsize=18,color='black',backgroundcolor='w')
ax2.text(78,603, 'C',fontsize=18,color='black',backgroundcolor='w')
plt.tight_layout()
plt.savefig(r"C:\workspace\GLCM\Neigiborhood_Analysis\Output\dist_dwn_stream_map.png",dpi=600)


    
    