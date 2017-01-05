# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 12:23:41 2016

@author: dan
"""

import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import numpy as np
from scikits.bootstrap import bootstrap as boot
import pytablewriter



def centeroidnp(df,query1,metric):
    length = df.query(query1)['homo_'+ metric].dropna().values.size
    sum_x = np.nansum(df.query(query1)['homo_'+ metric].values)
    sum_y = np.nansum(df.query(query1)['entropy_' +  metric].values)
    return sum_x/length, sum_y/length
    
def error_bars(df,query1,metric):
    y_err = np.nanstd(df.query(query1)['entropy_' + metric].values)
    x_err = np.nanstd(df.query(query1)['homo_'+ metric].values)
    return x_err, y_err

root = r'C:\workspace\GLCM\new_output'
files = glob(root + os.sep + '*zonal_stats_merged.csv')

homo_list = files[6:7]
entropy_list = files[5:6]
var_list = files[8:9]

n=0
    
homo = homo_list[n]
entropy = entropy_list[n]
var = var_list[n]
meter = homo.split('\\')[-1].split('_')[1]

df1 = pd.read_csv(homo,sep=',')
df2 = pd.read_csv(entropy,sep=',')
df3 = pd.read_csv(var,sep=',')
df1.rename(columns={'max':'homo_max', 'mean':'homo_mean', 'median':'homo_median','min':'homo_min','percentile_25':'homo_25','percentile_50':'homo_50', 'percentile_75':'homo_75','std':'homo_std'},inplace=True)   
df2.rename(columns={'max':'entropy_max', 'mean':'entropy_mean', 'median':'entropy_median','min':'entropy_min','percentile_25':'entropy_25','percentile_50':'entropy_50', 'percentile_75':'entropy_75','std':'entropy_std'},inplace=True)   
df3.rename(columns={'max':'var_max', 'mean':'var_mean', 'median':'var_median','min':'var_min','percentile_25':'var_25','percentile_50':'var_50', 'percentile_75':'var_75','std':'var_std'},inplace=True)   


merge =df1.merge(df2,left_index=True, right_index=True, how='left')
merge = merge.merge(df3,left_index=True, right_index=True, how='left' )
merge = merge[['homo_median','entropy_median','var_median','substrate']].dropna()
merge.rename(columns={'substrate_x':'substrate'},inplace=True)
del df1,df2,df3
grouped = merge.groupby('substrate')
sand = grouped.get_group('sand')
gravel = grouped.get_group('gravel')
boulders = grouped.get_group('boulders')


boot.ci(sand['homo_median'],np.median,alpha=0.05)
boot.ci(sand['entropy_median'],np.median,alpha=0.05)
boot.ci(sand['var_median'],np.median,alpha=0.05)


boot.ci(gravel['homo_median'],np.median,alpha=0.05)
boot.ci(gravel['entropy_median'],np.median,alpha=0.05)
boot.ci(gravel['var_median'],np.median,alpha=0.05)


boot.ci(boulders['homo_median'],np.median,alpha=0.05)
boot.ci(boulders['entropy_median'],np.median,alpha=0.05)
boot.ci(boulders['var_median'],np.median,alpha=0.05)

homo_df= pd.DataFrame(index=['Sand','Gravel','Boulders'], columns=['lbound','ubound'])
homo_df.loc['Sand'] = pd.Series({'lbound':boot.ci(sand['homo_median'],np.median,alpha=0.05)[0],'ubound': boot.ci(sand['homo_median'],np.median,alpha=0.05)[1]})
homo_df.loc['Gravel'] = pd.Series({'lbound':boot.ci(gravel['homo_median'],np.median,alpha=0.05)[0],'ubound': boot.ci(gravel['homo_median'],np.median,alpha=0.05)[1]})
homo_df.loc['Boulders'] = pd.Series({'lbound':boot.ci(boulders['homo_median'],np.median,alpha=0.05)[0],'ubound': boot.ci(boulders['homo_median'],np.median,alpha=0.05)[1]})
homo_df['mean']= homo_df['lbound'] + (homo_df['ubound']-homo_df['lbound'])/2
homo_df['Margin_of_Error'] = (homo_df['ubound']-homo_df['lbound'])/2
homo_df = homo_df[['mean','Margin_of_Error']].reset_index()
homo_df['plot_num'] = [0,1,2]

writer = pytablewriter.MarkdownTableWriter()
writer.table_name = "Homogeneity Table"
writer.header_list = ["substrate","mean","margin of error"]
writer.value_matrix = homo_df.values.tolist()
table = writer.write_table()


ent_df= pd.DataFrame(index=['Sand','Gravel','Boulders'], columns=['lbound','ubound'])
ent_df.loc['Sand'] = pd.Series({'lbound':boot.ci(sand['entropy_median'],np.median,alpha=0.05)[0],'ubound': boot.ci(sand['entropy_median'],np.median,alpha=0.05)[1]})
ent_df.loc['Gravel'] = pd.Series({'lbound':boot.ci(gravel['entropy_median'],np.median,alpha=0.05)[0],'ubound': boot.ci(gravel['entropy_median'],np.median,alpha=0.05)[1]})
ent_df.loc['Boulders'] = pd.Series({'lbound':boot.ci(boulders['entropy_median'],np.median,alpha=0.05)[0],'ubound': boot.ci(boulders['entropy_median'],np.median,alpha=0.05)[1]})
ent_df['mean']= ent_df['lbound'] + (ent_df['ubound']-ent_df['lbound'])/2
ent_df['Margin_of_Error'] = (ent_df['ubound']-ent_df['lbound'])/2
ent_df = ent_df[['mean','Margin_of_Error']].reset_index()
ent_df['plot_num'] = [0,1,2]

writer = pytablewriter.MarkdownTableWriter()
writer.table_name = "Entropy Table"
writer.header_list = ["substrate","mean","margin of error"]
writer.value_matrix = ent_df.values.tolist()
table = writer.write_table()


var_df= pd.DataFrame(index=['Sand','Gravel','Boulders'], columns=['lbound','ubound'])
var_df.loc['Sand'] = pd.Series({'lbound':boot.ci(sand['var_median'],np.median,alpha=0.05)[0],'ubound': boot.ci(sand['var_median'],np.median,alpha=0.05)[1]})
var_df.loc['Gravel'] = pd.Series({'lbound':boot.ci(gravel['var_median'],np.median,alpha=0.05)[0],'ubound': boot.ci(gravel['var_median'],np.median,alpha=0.05)[1]})
var_df.loc['Boulders'] = pd.Series({'lbound':boot.ci(boulders['var_median'],np.median,alpha=0.05)[0],'ubound': boot.ci(boulders['var_median'],np.median,alpha=0.05)[1]})
var_df['mean']= var_df['lbound'] + (var_df['ubound']-var_df['lbound'])/2
var_df['Margin_of_Error'] = (var_df['ubound']-var_df['lbound'])/2
var_df = var_df[['mean','Margin_of_Error']].reset_index()
var_df['plot_num'] = [0,1,2]

writer = pytablewriter.MarkdownTableWriter()
writer.table_name = "GLCM Variance Table"
writer.header_list = ["substrate","mean","margin of error"]
writer.value_matrix = var_df.values.tolist()
table = writer.write_table()





fig,(ax,ax1,ax2) = plt.subplots(ncols=3)
homo_df['mean'].plot(ax = ax, xticks=homo_df.index, yerr=homo_df['Margin_of_Error'],use_index=True,xlim=(-0.5,2.5),linestyle=' ',marker='o')
ax.set_ylabel('Homogeneity')
ax.set_xticklabels(homo_df['index'].values)

ent_df['mean'].plot(ax = ax1, xticks=ent_df.index, yerr=ent_df['Margin_of_Error'],use_index=True,xlim=(-0.5,2.5),linestyle=' ',marker='o')
ax1.set_ylabel('Entropy')
ax1.set_xticklabels(ent_df['index'].values)

var_df['mean'].plot(ax = ax2, xticks=var_df.index, yerr=var_df['Margin_of_Error'],use_index=True,xlim=(-0.5,2.5),linestyle=' ',marker='o')
ax2.set_ylabel('GLCM Variance')
ax2.set_xticklabels(var_df['index'].values)
plt.suptitle('GLCM Substrate Characteristics ')
plt.tight_layout()
