# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 16:23:42 2016

@author: dan
"""

import pandas as pd
from glob import glob
import os
import sys

#root = sys.argv[1]
root = r'C:\workspace\GLCM\d_5_and_angle_0'
files_04 = glob(r"C:\workspace\GLCM\output\2014_04\*zonalstats*csv")
files_09 = glob(r"C:\workspace\GLCM\output\2014_09\*zonalstats*csv")
files_09_2 = glob(r"C:\workspace\GLCM\output\2014_09_2\*zonalstats*csv")
n = 0
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
    meter = file_04.split('\\')[-1].split('.')[0].split('_')[-2] 
   
    
    oName = root + os.sep + variable + "_" + meter + "_zonal_stats_merged.csv"
    merge.rename(columns={'glcm_mean':variable + '_mean'},inplace=True)
    merge.to_csv(oName, sep=',', index=False)
    