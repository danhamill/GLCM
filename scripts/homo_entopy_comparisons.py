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

files = glob(r"C:\workspace\GLCM\output\*csv")

homo_list = files[30:35]
entropy_list = files[25:30]

for n in xrange(len(homo_list)):
    
    homo = homo_list[n]
    entropy = entropy_list[n]
    meter = homo.split('\\')[-1].split('_')[1]
    
    df1 = pd.read_csv(homo,sep=',')
    df2 = pd.read_csv(entropy,sep=',')
    
    merge =df1.merge(df2,left_index=True, right_index=True, how='left')
    
    merge = merge[['homo_mean','entropy_mean','substrate_x']]
    merge.rename(columns={'substrate_x':'substrate'},inplace=True)
    
    #legend stuff
    blue = mpatches.Patch(color='blue',label='Sand')
    green = mpatches.Patch(color='green',label='Gravel')
    red = mpatches.Patch(color='red',label='Boulders')

    
    oName = r"C:\workspace\GLCM\output\2014_04" + os.sep + "homo_entropy_comparison_" + meter +".png"   
    fig,ax = plt.subplots()

    merge.query('substrate == "sand"').plot.scatter(ax =ax, x='homo_mean',y='entropy_mean',color='blue')
    merge.query('substrate == "gravel"').plot.scatter(ax =ax, x='homo_mean',y='entropy_mean',color='green')
    merge.query('substrate == "boulders"').plot.scatter(ax =ax, x='homo_mean',y='entropy_mean',color='red')
    ax.legend(loc=9,handles=[blue,green,red],ncol=3,columnspacing=1, fontsize=8)
    plt.suptitle(meter + ' meter grid')
    plt.tight_layout()
    plt.savefig(oName,dpi=600)