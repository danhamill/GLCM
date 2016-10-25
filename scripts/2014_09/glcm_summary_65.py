# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 16:43:23 2016

@author: dan
"""


from rasterstats import zonal_stats
import ogr
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import table
import os
import numpy as np
import matplotlib.patches as mpatches

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
    
def agg_distributions(stats,in_shp):
    #Lets get get the substrate to sort lists
    ds = ogr.Open(in_shp)
    lyr = ds.GetLayer(0)
    a=[]
    for row in lyr:
        a.append(row.substrate)
    lyr.ResetReading()
    del ds

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

    s_df = make_df2(s)
    g_df = make_df2(g)
    r_df = make_df2(b)
    del s,  g,  b
    return s_df,  g_df, r_df,a
    
def make_table(s_df, g_df, b_df):
    tbl = pd.DataFrame(columns=['substrate','10%','20%','25%','50%','75%','kurt','skew'])
    tbl['substrate']=['sand','gravel','boulders']
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
        tbl.loc['boulders'] = pd.Series({'10%':float(b_df.quantile(0.1).values),'20%':float(b_df.quantile(0.2).values),'25%':float(b_df.describe().iloc[4].values), '50%':float(b_df.describe().iloc[5].values),'75%':float(b_df.describe().iloc[6].values),'kurt':float(b_df.kurtosis().values),'skew':float(b_df.skew().values)})
    except:
        tbl.loc['boulders'] = pd.Series({'10%':np.nan,'20%':np.nan,'25%':np.nan, '50%':np.nan,'75%':np.nan,'kurt':np.nan,'skew':np.nan})
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
        ss_raster = r"C:\workspace\Merged_SS\raster\2014_09\ss_2014_09_R01765_raster.tif"
        in_shp = r"C:\workspace\Merged_SS\window_analysis\shapefiles\R01765.shp"
        
        contFile = r"C:\workspace\GLCM\output\glcm_rasters\2014_09_2" + os.sep + meter +os.sep+"R01765_" + meter + "_contrast.tif"
        dissFile = r"C:\workspace\GLCM\output\glcm_rasters\2014_09_2" + os.sep + meter +os.sep+"R01765_" + meter + "_diss.tif"
        homoFile = r"C:\workspace\GLCM\output\glcm_rasters\2014_09_2" + os.sep + meter +os.sep+"R01765_" + meter + "_homo.tif"
        energyFile = r"C:\workspace\GLCM\output\glcm_rasters\2014_09_2" + os.sep + meter +os.sep+"R01765_" + meter + "_energy.tif"
        corrFile = r"C:\workspace\GLCM\output\glcm_rasters\2014_09_2" + os.sep + meter +os.sep+"R01765_" + meter + "_corr.tif"
        ASMFile = r"C:\workspace\GLCM\output\glcm_rasters\2014_09_2" + os.sep + meter +os.sep+"R01765_" + meter + "_asm.tif"    
        ENTFile = r"C:\workspace\GLCM\output\glcm_rasters\2014_09_2" + os.sep + meter +os.sep+"R01765_" + meter + "_entropy.tif"
        
        ss_stats = zonal_stats(in_shp, ss_raster, stats=['count','mean','std'])
        ss_df = pd.DataFrame(ss_stats)
        ss_df.rename(columns={'count':'ss_count','mean':'ss_mean','std':'ss_std'},inplace=True)

        
        raster_list = [contFile, dissFile, homoFile, energyFile, corrFile, ASMFile,ENTFile]
        
        for raster in raster_list[:]:
            
            print 'now working on %s ....' %(raster,)
            variable = raster.split('\\')[-1].split('.')[0].split('_')[-1]
            
            z_stats = zonal_stats(in_shp, raster, stats=['count','mean'], raster_out=True)
            
            s_df,  g_df, r_df, a = agg_distributions(z_stats, in_shp)
            
            #Create Summary Table
            agg_tbl = make_table(s_df, g_df, r_df)
            oName = r"C:\workspace\GLCM\output\2014_09_2" + os.sep + variable + "_aggragrated_" + meter +"_distribution.csv"
            agg_tbl.to_csv(oName,sep=',')
            
            oName = r"C:\workspace\GLCM\output\2014_09_2" + os.sep + variable + "_aggragrated_" + meter +".png"

            plot_agg_table(agg_tbl,oName, meter)
            
            oName = r"C:\workspace\GLCM\output\2014_09_2" + os.sep + variable + "_aggragrated_" + meter +"_distribution.png"
            
            
            #legend stuff
            blue = mpatches.Patch(color='blue',label='Sand')
            green = mpatches.Patch(color='green',label='Gravel')
            red = mpatches.Patch(color='red',label='Boulders')
            
            fig = plt.figure(figsize=(6,2))
            ax = fig.add_subplot(1,1,1)
            try:
                s_df.plot.hist(ax=ax,bins=50,legend=False,rot=45,zorder=1,color='blue')            
            except:
                pass
                    
            #ax = fig.add_subplot(1,3,2)
            try:
               g_df.plot.hist(ax=ax,bins=50,legend=False,rot=45,zorder=1, color='green')
            except:
                pass            
            #ax = fig.add_subplot(1,3,3)
            try:
                  r_df.plot.hist(ax=ax,bins=50,legend=False,rot=45,zorder=10, color='red')
            except: 
                pass
            
            ax.set_xlabel(variable)            
            ax.legend(loc=9,handles=[blue,green,red],ncol=3,columnspacing=1, fontsize=8)
            plt.suptitle( meter + ' meter grid')
            plt.tight_layout(pad=2)            
            plt.savefig(oName, dpi=600)
            
            oName = r"C:\workspace\GLCM\output\2014_09_2" + os.sep + variable + "_zonalstats_" + meter +"_grid.csv" 
            glcm_stats = zonal_stats(in_shp, raster, stats=['min','mean','max','median','std','count','percentile_25','percentile_50','percentile_75'])
            glcm_df = pd.DataFrame(glcm_stats)
            glcm_df['substrate'] = a
            glcm_df.to_csv(oName,sep=',',index=False)
            
            
            glcm_stats = zonal_stats(in_shp, raster, stats=['count','mean'])   

            glcm_df = pd.DataFrame(glcm_stats)
            glcm_df.rename(columns={'count':'glcm_count','mean':'glcm_mean'},inplace = True)
            merge = glcm_df.merge(ss_df, left_index=True, right_index=True, how='left')
            
            merge['substrate'] = a

            
            oName = r"C:\workspace\GLCM\output\2014_09_2" + os.sep + variable + "_ss_comparison_" + meter +".csv"   
            merge.to_csv(oName, sep=',',index=False)
            
            oName = r"C:\workspace\GLCM\output\2014_09_2" + os.sep + variable + "_ss_comparison_" + meter +".png"
            fig, (ax1,ax2) = plt.subplots(nrows=2)

            
            try:
                merge.query('substrate == ["sand"]').plot.scatter(ax = ax1, x='ss_mean', y='glcm_mean', color='blue',label='sand')
            except:                
                pass
            try:
                merge.query('substrate == ["gravel"]').plot.scatter(ax = ax1, x='ss_mean', y='glcm_mean', color='red',label = 'gravel')
            except:
                pass
            try:
                merge.query('substrate == ["boulders"]').plot.scatter(ax = ax1, x='ss_mean', y='glcm_mean', color='green', label='boulders')
            except:
                pass
            ax1.set_ylabel(variable)
            ax1.legend(loc='9', ncol=3, columnspacing=1, fontsize=8)
           
            try:
                merge.query('substrate == ["sand"]').plot.scatter(ax = ax2, x='ss_std', y='glcm_mean', color='blue',label='sand')
            except:
                pass
            try:
                merge.query('substrate == ["gravel"]').plot.scatter(ax = ax2, x='ss_std', y='glcm_mean', color='red',label = 'gravel')
            except:
                pass
            try:
                merge.query('substrate == ["boulders"]').plot.scatter(ax = ax2, x='ss_std', y='glcm_mean', color='green', label='boulders')
            except:
                pass
            ax2.set_ylabel(variable)
            ax2.legend(loc='9', ncol=3, columnspacing=1, fontsize=8)
            plt.tight_layout()
            plt.suptitle(meter +' meter grid')
            plt.savefig(oName,dpi=600)
            plt.close()








