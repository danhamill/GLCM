# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 16:23:42 2016

@author: dan
"""

import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import os


files_04 = glob(r"C:\workspace\GLCM\output\2014_04\*csv")
files_09 = glob(r"C:\workspace\GLCM\output\2014_09\*csv")
files_09_2 = glob(r"C:\workspace\GLCM\output\2014_09_2\*csv")

for n in xrange(len(files_04)):
    #print n
    file_04 = files_04[n]
    file_09 = files_09[n]
    file_09_2 = files_09_2[n]
    

    df1 = pd.read_csv(file_04, sep=',')
    df2 = pd.read_csv(file_09, sep=',')
    df3 = pd.read_csv(file_09_2, sep=',')
    merge = pd.concat([df1,df2,df3])   
    del df1,df2, df3
    
    variable = file_04.split('\\')[-1].split('_')[0]
    meter = file_04.split('\\')[-1].split('.')[0].split('_')[-1] 
    
    oName = r"C:\workspace\GLCM\output" + os.sep + variable + "_" + meter + "_combined.png"
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
    ax2.legend(loc=9, ncol=3, columnspacing=1, fontsize=8)
    plt.tight_layout()
    plt.suptitle(meter +' meter grid')
    plt.savefig(oName,dpi=600)    
    
    oName = r"C:\workspace\GLCM\output" + os.sep + variable + "_" + meter + "_combined.csv"
    merge.rename(columns={'glcm_mean':variable + '_mean'},inplace=True)
    merge.to_csv(oName, sep=',', index=False)
    