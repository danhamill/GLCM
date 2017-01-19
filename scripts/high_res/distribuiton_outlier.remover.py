# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 15:53:28 2017

@author: dan
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_distributions(merge_dist):
    #legend stuff
    blue = mpatches.Patch(color='blue',label='Sand')
    green = mpatches.Patch(color='green',label='Gravel')
    red = mpatches.Patch(color='red',label='Boulders')
    
    fig, [ax,ax1,ax2] = plt.subplots(nrows=3)
    merge_dist.groupby('sedclass')['Entropy'].plot.hist(ax=ax,bins=50,normed=True)    
    ax.set_xlabel('Entropy')
    ax.legend(loc=9,handles=[blue,green,red],ncol=3,columnspacing=1, fontsize=8)
    merge_dist.groupby('sedclass')['Homogeneity'].plot.hist(ax=ax1,bins=50,normed=True)    
    ax1.set_xlabel('Homogeneity')
    ax1.legend(loc=9,handles=[blue,green,red],ncol=3,columnspacing=1, fontsize=8)
    merge_dist.groupby('sedclass')['Variance'].plot.hist(ax=ax2,bins=50,normed=True)         
    ax2.set_xlabel('GLCM Variance')
    ax2.legend(loc=9,handles=[blue,green,red],ncol=3,columnspacing=1, fontsize=8)
    plt.tight_layout()
    plt.show()
    
    
    
merge_dist= pd.read_csv(r"C:\workspace\GLCM\new_output\merged_aggregraded_distributions.csv",sep=',')


merge_dist = merge_dist[(merge_dist['Entropy']>2.5) & (merge_dist['Variance']<12)&(merge_dist['Entropy']<4.6) ]
                        
merge_dist.to_csv(r"C:\workspace\GLCM\new_output\merged_aggregraded_distributions_no_outlier.csv",sep=',',index=False)
