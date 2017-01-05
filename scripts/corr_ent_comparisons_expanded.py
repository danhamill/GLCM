# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 17:51:38 2016

@author: dan
"""

import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import numpy as np
from pandas.tools.plotting import table
import sys

def centeroidnp(df,query1,metric):
    length = df.query(query1)['corr_'+ metric].dropna().values.size
    sum_x = np.nansum(df.query(query1)['corr_'+ metric].values)
    sum_y = np.nansum(df.query(query1)['var_' +  metric].values)
    return sum_x/length, sum_y/length
    
def error_bars(df,query1,metric):
    y_err = np.nanstd(df.query(query1)['var_' + metric].values)
    x_err = np.nanstd(df.query(query1)['corr_'+ metric].values)
    return x_err, y_err
    
#root = sys.argv[1]
root = 'C:\\workspace\\GLCM\\d_5_and_angle_0'
files = glob(root + os.sep + '*.csv')


corr_list = files[10:15]
var_list = files[25:30]

print "will save files to root directory %s" %(root,)

n=3
for n in xrange(len(corr_list)):
    
    homo = corr_list[n]
    var = var_list[n]
    meter = homo.split('\\')[-1].split('_')[1]
    
    df1 = pd.read_csv(homo,sep=',')
    df2 = pd.read_csv(var,sep=',')
    df1.rename(columns={'max':'corr_max', 'mean':'corr_mean', 'median':'corr_median','min':'corr_min','percentile_25':'corr_25','percentile_50':'corr_50', 'percentile_75':'corr_75','std':'corr_std'},inplace=True)   
    df2.rename(columns={'max':'var_max', 'mean':'var_mean', 'median':'var_median','min':'var_min','percentile_25':'var_25','percentile_50':'var_50', 'percentile_75':'var_75','std':'var_std'},inplace=True)   
    
    
    merge =df1.merge(df2,left_index=True, right_index=True, how='left')
    
    merge = merge[['corr_mean','var_mean','substrate_x','corr_25','var_25','corr_75','var_75']]
    merge.rename(columns={'substrate_x':'substrate'},inplace=True)
    
    #Queries
    s_query = 'substrate == "sand"'
    g_query = 'substrate == "gravel"'
    b_query = 'substrate == "boulders"'
    
        
    #25percent stuff
    #Centroid stuff
    s_centroid = centeroidnp(merge,s_query,str(25))    
    g_centroid = centeroidnp(merge,g_query,str(25))
    b_centroid = centeroidnp(merge,b_query,str(25))
    
    
    
    #errorbars
    s_err = error_bars(merge,s_query,str(25))    
    g_err = error_bars(merge,g_query,str(25))
    b_err = error_bars(merge,b_query,str(25))
    
    cent_df = pd.DataFrame(columns=['x_cent','y_cent','y_err','x_err'], index=['sand','gravel','boulders'])
    cent_df.loc['sand'] = pd.Series({'x_cent':s_centroid[0] ,'y_cent':s_centroid[1] ,'y_err':s_err[1] ,'x_err':s_err[0]})
    cent_df.loc['gravel'] = pd.Series({'x_cent':g_centroid[0] ,'y_cent': g_centroid[1],'y_err':g_err[1] ,'x_err':g_err[0]})
    cent_df.loc['boulders'] = pd.Series({'x_cent':b_centroid[0] ,'y_cent':b_centroid[1] ,'y_err': b_err[1],'x_err':b_err[0]})
    cent_df = cent_df.reset_index()
    
    #legend stuff
    blue = mpatches.Patch(color='blue',label='Sand')
    green = mpatches.Patch(color='green',label='Gravel')
    red = mpatches.Patch(color='red',label='Boulders')
    
    
    oName = root + os.sep + "corr_variance_comparison_expanded_" + meter +".png"   
    fig,axes = plt.subplots(figsize =(8,6),nrows=3,ncols=2)
    
    merge.query('substrate == "sand"').plot.scatter(ax =axes[0,0], x='corr_25',y='var_25',color='blue')
    cent_df.query('index == "sand"').plot.scatter(ax = axes[0,0], x='x_cent',y='y_cent', yerr= 'y_err', xerr='x_err',color='blue',s=100)
    merge.query('substrate == "gravel"').plot.scatter(ax =axes[0,0], x='corr_25',y='var_25',color='green')
    cent_df.query('index == "gravel"').plot.scatter(ax = axes[0,0], x='x_cent',y='y_cent', yerr= 'y_err', xerr='x_err',color='green',s=100)
    cent_df.query('index == "boulders"').plot.scatter(ax = axes[0,0], x='x_cent',y='y_cent', yerr= 'y_err', xerr='x_err',color='red',s=100)
    merge.query('substrate == "boulders"').plot.scatter(ax =axes[0,0],x='corr_25',y='var_25',color='red')
    axes[0,0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0., handles=[blue,green,red], ncol=3, columnspacing=1, fontsize=8)
    
    axes[0,1].xaxis.set_visible(False)
    axes[0,1].yaxis.set_visible(False)
    #hide the spines
    for sp in axes[0,1].spines.itervalues():
        sp.set_color('w')
        sp.set_zorder(0)
    the_table = table(axes[0,1], cent_df.set_index('index').applymap(lambda x: round(x,3)),loc='center',colWidths=[0.1,0.1,0.1,0.1],zorder=0)
    # mean stuff
    s_centroid = centeroidnp(merge,s_query,'mean')    
    g_centroid = centeroidnp(merge,g_query,'mean')
    b_centroid = centeroidnp(merge,b_query,'mean')
    
    s_err = error_bars(merge,s_query,'mean')    
    g_err = error_bars(merge,g_query,'mean')
    b_err = error_bars(merge,b_query,'mean')
    
    cent_df = pd.DataFrame(columns=['x_cent','y_cent','y_err','x_err'], index=['sand','gravel','boulders'])
    cent_df.loc['sand'] = pd.Series({'x_cent':s_centroid[0] ,'y_cent':s_centroid[1] ,'y_err':s_err[1] ,'x_err':s_err[0]})
    cent_df.loc['gravel'] = pd.Series({'x_cent':g_centroid[0] ,'y_cent': g_centroid[1],'y_err':g_err[1] ,'x_err':g_err[0]})
    cent_df.loc['boulders'] = pd.Series({'x_cent':b_centroid[0] ,'y_cent':b_centroid[1] ,'y_err': b_err[1],'x_err':b_err[0]})
    cent_df = cent_df.reset_index()
    merge.query('substrate == "sand"').plot.scatter(ax =axes[1,0], x='corr_mean',y='var_mean',color='blue')
    cent_df.query('index == "sand"').plot.scatter(ax = axes[1,0], x='x_cent',y='y_cent', yerr= 'y_err', xerr='x_err',color='blue',s=100)
    merge.query('substrate == "gravel"').plot.scatter(ax =axes[1,0], x='corr_mean',y='var_mean',color='green')
    cent_df.query('index == "gravel"').plot.scatter(ax = axes[1,0], x='x_cent',y='y_cent', yerr= 'y_err', xerr='x_err',color='green',s=100)
    cent_df.query('index == "boulders"').plot.scatter(ax = axes[1,0], x='x_cent',y='y_cent', yerr= 'y_err', xerr='x_err',color='red',s=100)
    merge.query('substrate == "boulders"').plot.scatter(ax =axes[1,0],x='corr_mean',y='var_mean',color='red')
    #axes[1,0].legend(loc=9,handles=[blue,green,red],ncol=3,columnspacing=1, fontsize=8)
    
    axes[1,1].xaxis.set_visible(False)
    axes[1,1].yaxis.set_visible(False)
    #hide the spines
    for sp in axes[1,1].spines.itervalues():
        sp.set_color('w')
        sp.set_zorder(0)    
    the_table = table(axes[1,1], cent_df.set_index('index').applymap(lambda x: round(x,3)),loc='center',colWidths=[0.1,0.1,0.1,0.1])
    
    # 75 percent stuff
    s_centroid = centeroidnp(merge,s_query,str(75))    
    g_centroid = centeroidnp(merge,g_query,str(75))
    b_centroid = centeroidnp(merge,b_query,str(75))
    
    s_err = error_bars(merge,s_query,str(75))    
    g_err = error_bars(merge,g_query,str(75))
    b_err = error_bars(merge,b_query,str(75))
    
    cent_df = pd.DataFrame(columns=['x_cent','y_cent','y_err','x_err'], index=['sand','gravel','boulders'])
    cent_df.loc['sand'] = pd.Series({'x_cent':s_centroid[0] ,'y_cent':s_centroid[1] ,'y_err':s_err[1] ,'x_err':s_err[0]})
    cent_df.loc['gravel'] = pd.Series({'x_cent':g_centroid[0] ,'y_cent': g_centroid[1],'y_err':g_err[1] ,'x_err':g_err[0]})
    cent_df.loc['boulders'] = pd.Series({'x_cent':b_centroid[0] ,'y_cent':b_centroid[1] ,'y_err': b_err[1],'x_err':b_err[0]})
    
    cent_df = cent_df.reset_index()
    merge.query('substrate == "sand"').plot.scatter(ax =axes[2,0], x='corr_75',y='var_75',color='blue')
    cent_df.query('index == "sand"').plot.scatter(ax = axes[2,0], x='x_cent',y='y_cent', yerr= 'y_err', xerr='x_err',color='blue',s=100)
    
    merge.query('substrate == "gravel"').plot.scatter(ax =axes[2,0], x='corr_75',y='var_75',color='green')
    cent_df.query('index == "gravel"').plot.scatter(ax = axes[2,0], x='x_cent',y='y_cent', yerr= 'y_err', xerr='x_err',color='green',s=100)
    cent_df.query('index == "boulders"').plot.scatter(ax = axes[2,0], x='x_cent',y='y_cent', yerr= 'y_err', xerr='x_err',color='red',s=100)
    merge.query('substrate == "boulders"').plot.scatter(ax =axes[2,0],x='corr_75',y='var_75',color='red')
    
    #axes[2,0].legend(loc=9,handles=[blue,green,red],ncol=3,columnspacing=1, fontsize=8)
    
    axes[2,1].xaxis.set_visible(False)
    axes[2,1].yaxis.set_visible(False)
    #hide the spines
    for sp in axes[2,1].spines.itervalues():
        sp.set_color('w')
        sp.set_zorder(0)
    the_table = table(axes[2,1], cent_df.set_index('index').applymap(lambda x: round(x,3)),loc='center',colWidths=[0.1,0.1,0.1,0.1])
    
    plt.suptitle(meter + ' meter grid')
    plt.tight_layout(pad=2)
    plt.savefig(oName,dpi=600)
    
    del merge, cent_df,df1,df2,s_centroid,g_centroid,b_centroid,s_err,b_err,g_err
    