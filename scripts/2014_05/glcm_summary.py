# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 16:43:23 2016

@author: dan
"""


from rasterstats import zonal_stats
import gdal
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import table
import os
import numpy as np


def assign_class(row):
    if row.sed5class == 1:
        return 'sand'
    if row.sed5class == 2:
        return 'sand/gravel'
    if row.sed5class == 3:
        return 'gravel'
    if row.sed5class == 4:
        return 'sand/rock'
    if row.sed5class == 5:
        return 'rock'
        
def make_df2(x):
    df = pd.DataFrame(x,columns=['dBW'])
    return df
    
def agg_distributions(stats, sed_pts, ss_stats):
    pts_df = pd.read_csv(sed_pts,sep=',',names=['northing','easting','sed5class'])
    pts_df['sed5class'] = pts_df.apply(lambda row: assign_class(row), axis = 1)
    
    bound = max(stat['count'] for stat in stats)
    s, sg, g, sr, r = [],[],[],[],[]
    n = 0
    for item in stats:
        raster_array = item['mini_raster_array'].compressed()
        substrate = pts_df['sed5class'][n]
        mean = ss_stats[n]['mean']
        if substrate=='sand' and item['count']==bound and mean < 20:
            s.extend(list(raster_array))
        if substrate=='sand/gravel' and item['count'] and mean < 20:
            sg.extend(list(raster_array))
        if substrate=='gravel' and item['count']==bound and mean < 20:
            g.extend(list(raster_array))
        if substrate=='sand/rock' and item['count']==bound and mean < 20:
            sr.extend(list(raster_array))
        if substrate=='rock' and item['count']==bound and mean < 20:
            r.extend(list(raster_array))
        n+=1        
    del raster_array, substrate, n, item, bound

    s_df = make_df2(s)
    sg_df = make_df2(sg)
    g_df = make_df2(g)
    sr_df = make_df2(sr)
    r_df = make_df2(r)
    del s, sg, g, sr, r
    return s_df, sg_df, g_df, sr_df, r_df
    
def make_table(s_df, g_df, b_df):
    tbl = pd.DataFrame(columns=['substrate','10%','20%','25%','50%','75%','kurt','skew'])
    tbl['substrate']=['sand','gravel','rock']
    tbl = tbl.set_index('substrate')
    
    try:
        tbl.loc['sand'] = pd.Series({'10%':float(s_df.quantile(0.1).values),'20%':float(s_df.quantile(0.2).values),'25%':float(s_df.describe().iloc[4].values), '50%':float(s_df.describe().iloc[5].values),'75%':float(s_df.describe().iloc[6].values),'kurt':float(s_df.kurtosis().values),'skew':float(s_df.skew().values)})
    except:
        tbl.loc['sand'] = pd.Series({'10%':np.nan,'20%':np.nan,'25%':np.nan, '50%':np.nan,'75%':np.nan,'kurt':np.nan,'skew':np.nan})
    try:
        tbl.loc['gravel'] = pd.Series({'10%':float(g_df.quantile(0.1).values),'20%':float(g_df.quantile(0.2).values),'25%':float(g_df.describe().iloc[4].values), '50%':float(g_df.describe().iloc[5].values),'75%':float(g_df.describe().iloc[6].values),'kurt':float(g_df.kurtosis().values),'skew':float(g_df.skew().values)})
    except:
        tbl.loc['gravel'] = pd.Series({'10%':np.nan,'20%':np.nan,'25%':np.nan, '50%':np.nan,'75%':np.nan,'kurt':np.nan,'skew':np.nan})
    try:        
        tbl.loc['rock'] = pd.Series({'10%':float(b_df.quantile(0.1).values),'20%':float(b_df.quantile(0.2).values),'25%':float(b_df.describe().iloc[4].values), '50%':float(b_df.describe().iloc[5].values),'75%':float(b_df.describe().iloc[6].values),'kurt':float(b_df.kurtosis().values),'skew':float(b_df.skew().values)})
    except:
        tbl.loc['rock'] = pd.Series({'10%':np.nan,'20%':np.nan,'25%':np.nan, '50%':np.nan,'75%':np.nan,'kurt':np.nan,'skew':np.nan})
    tbl = tbl.applymap(lambda x: round(x,3))
    return tbl
    
def plot_agg_table(agg_tbl,oName,meter):
    fig = plt.figure(figsize=(6,1.5))
    ax2 = fig.add_subplot(111)
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    for sp in ax2.spines.itervalues():
        sp.set_color('w')
        sp.set_zorder(0)
    the_table = table(ax2, agg_tbl ,loc='upper center',colWidths=[0.1,0.1,0.1,0.1,0.1,0.1,0.1])
    the_table.set_fontsize(10)
    plt.suptitle(meter +' meter grid')
    plt.tight_layout()
    plt.savefig(oName, dpi = 600)
    
    
if __name__ == '__main__':  
    win_sizes = [8,12,20,40,80]
    for win_size in win_sizes[:]:   
        win = win_size
        meter = str(win/4)
        print 'Now working on %s grid resolution...' %(meter,)
        ss_raster = r"C:\workspace\GLCM\input\2014_05\ss_R02028.tif"
        in_shp = r"C:\workspace\Reach_4a\Multibeam\mb_sed_class\output\shapefiles\may2014_" + meter + "m_buff.shp"
        contFile = r"C:\workspace\GLCM\output\glcm_rasters\2014_05" + os.sep + meter +os.sep+"R02028_" + meter + "_contrast.tif"
        dissFile = r"C:\workspace\GLCM\output\glcm_rasters\2014_05" + os.sep + meter +os.sep+"R02028_" + meter + "_diss.tif"
        homoFile = r"C:\workspace\GLCM\output\glcm_rasters\2014_05" + os.sep + meter +os.sep+"R02028_" + meter + "_homo.tif"
        energyFile = r"C:\workspace\GLCM\output\glcm_rasters\2014_05" + os.sep + meter +os.sep+"R02028_" + meter + "_energy.tif"
        corrFile = r"C:\workspace\GLCM\output\glcm_rasters\2014_05" + os.sep + meter +os.sep+"R02028_" + meter + "_corr.tif"
        ASMFile = r"C:\workspace\GLCM\output\glcm_rasters\2014_05" + os.sep + meter +os.sep+"R02028_" + meter + "_asm.tif"    
        sed_pts = r"C:\workspace\Reach_4a\Multibeam\mb_sed_class\may2014_mb6086r_sedclass\xy_sed5class_"+ meter + "m.csv"
        
        pts_df = pd.read_csv(sed_pts,sep=',',names=['northing','easting','sed5class'])
        pts_df['sed5class'] = pts_df.apply(lambda row: assign_class(row), axis = 1)
        
        ss_stats = zonal_stats(in_shp, ss_raster, stats=['count','mean','std'])
        print 'Number of ss stats lines is %s' %(str(len(ss_stats)),)
        ss_df = pd.DataFrame(ss_stats)
        
        
        raster_list = [contFile, dissFile, homoFile, energyFile, corrFile, ASMFile]
        
        for raster in raster_list[:]:
            
            print 'now working on %s ....' %(raster,)
            variable = raster.split('\\')[-1].split('.')[0].split('_')[-1]
            
            z_stats = zonal_stats(in_shp, raster, stats=['count','mean'], raster_out=True)
            
            s_df, sg_df, g_df, sr_df, r_df = agg_distributions(z_stats, sed_pts,ss_stats)
            
            print 'Length of sand data frame is %s' %(str(len(s_df)),)
            
            #Create Summary Table
            agg_tbl = make_table(s_df, g_df, r_df)

            oName = r"C:\workspace\GLCM\output\2014_05" + os.sep + variable + "_aggragrated_" + meter +".png"

            plot_agg_table(agg_tbl,oName, meter)
            
            oName = r"C:\workspace\GLCM\output\2014_05" + os.sep + variable + "_aggragrated_" + meter +"_distribution.png"
            
            fig = plt.figure(figsize=(6,2))
            ax = fig.add_subplot(1,3,1)
            try:
                s_df.plot.hist(ax=ax,bins=50,title='Sand',legend=False,rot=45)            
            except:
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                for sp in ax.spines.itervalues():
                    sp.set_color('w')
                    sp.set_zorder(0)
                    
            ax = fig.add_subplot(1,3,2)
            try:
               g_df.plot.hist(ax=ax,bins=50,title='Gravel',legend=False,rot=45)
            except:
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                for sp in ax.spines.itervalues():
                    sp.set_color('w')
                    sp.set_zorder(0)
            
            ax = fig.add_subplot(1,3,3)
            try:
                  r_df.plot.hist(ax=ax,bins=50,title='Rock',legend=False,rot=45)
            except: 
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                for sp in ax.spines.itervalues():
                    sp.set_color('w')
                    sp.set_zorder(0)
            plt.suptitle(variable + meter + ' meter grid')
            plt.tight_layout(pad=2)
            plt.savefig(oName, dpi=600)
            
            
            glcm_stats = zonal_stats(in_shp, raster, stats=['count','mean'])   
            bound = max(stat['count'] for stat in glcm_stats)
            glcm_df = pd.DataFrame(glcm_stats)
            
            merge = glcm_df.merge(ss_df, left_index=True, right_index=True, how='left')
            

            
            
            merge.rename(columns={'mean_x':'ss_mean','mean_y':'glcm_mean'},inplace = True)
            merge['sed5class'] = pts_df['sed5class']
            merge = merge[merge['count_x']==bound]
            
            oName = r"C:\workspace\GLCM\output\2014_04" + os.sep + variable + "_ss_comparison_" + meter +".csv"   
            merge.to_csv(oName, sep=',')

            oName = r"C:\workspace\GLCM\output\2014_05" + os.sep + variable + "_ss_comparison_" + meter +".png"
            fig, (ax1,ax2) = plt.subplots(nrows=2)
            #for name, group in groups:
            colors = {'sand': 'red', 'gravel': 'blue', 'rock': 'green'}
            
            try:
                merge.query('sed5class == ["sand"]').plot.scatter(ax = ax1, x='ss_mean', y='glcm_mean', color='blue',label='sand')
            except:                
                pass
            try:
                merge.query('sed5class == ["gravel"]').plot.scatter(ax = ax1, x='ss_mean', y='glcm_mean', color='red',label = 'gravel')
            except:
                pass
            try:
                merge.query('sed5class == ["rock"]').plot.scatter(ax = ax1, x='ss_mean', y='glcm_mean', color='green', label='rock')
            except:
                pass
            ax1.set_ylabel(variable)
           
            try:
                merge.query('sed5class == ["sand"]').plot.scatter(ax = ax2, x='std', y='glcm_mean', color='blue',label='sand')
            except:
                pass
            try:
                merge.query('sed5class == ["gravel"]').plot.scatter(ax = ax2, x='std', y='glcm_mean', color='red',label = 'gravel')
            except:
                pass
            try:
                merge.query('sed5class == ["rock"]').plot.scatter(ax = ax2, x='std', y='glcm_mean', color='green', label='rock')
            except:
                pass
            ax2.set_ylabel(variable)
            plt.tight_layout()
            plt.suptitle(meter +' grid')
            plt.savefig(oName,dpi=600)









