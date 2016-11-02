# -*- coding: utf-8 -*-
"""
Created on Tue Nov 01 16:03:01 2016

@author: dan
"""

import pandas as pd
import numpy as np
from __future__ import division #need this import for int32 dividion
from scipy.stats import variation

def centeroidnp(df,s_query,g_query,b_query,metric):
    s = np.nanmedian(df.query(s_query)[metric])
    g = np.nanmedian(df.query(g_query)[metric])
    b= np.nanmedian(df.query(b_query)[metric])
    return s,g,b

def find_weights(df):
    s_query = 'substrate == "sand"'
    g_query = 'substrate == "gravel"'
    b_query = 'substrate == "boulders"'
    
    #Find Standard Deviations
    sand_std = np.nanstd(df.query(s_query)['mean'].values)
    gravel_std = np.nanstd(df.query(g_query)['mean'].values)
    boulers_std = np.nanstd(df.query(b_query)['mean'].values)
    
    #Find Value counts
    sand_count = df.query(s_query)['mean'].dropna().size
    gravel_count = df.query(g_query)['mean'].dropna().size
    boulder_count = df.query(b_query)['mean'].dropna().size
    
    s,g,b =centeroidnp(df,s_query,g_query,b_query,'mean')
    #Find counts within +- 1std
    lbound = s - sand_std
    ubound = s + sand_std
    sand_threshold= sum((df.query(s_query)['mean']>lbound) & (df.query(s_query)['mean']<ubound))
    lbound = g - gravel_std
    ubound = g + gravel_std
    gravel_threshhold = sum((df.query(g_query)['mean']>lbound) & (df.query(g_query)['mean']<ubound))
    lbound = b - boulers_std
    ubound = b + boulers_std
    boulder_threshold =sum((df.query(b_query)['mean']>lbound) & (df.query(b_query)['mean']<ubound))
    
    #calculate weghts
    sand = sand_threshold/sand_count
    gravel = gravel_threshhold/gravel_count
    boulders = boulder_threshold/boulder_count
    print sand, gravel, boulders
    return np.average(sand,gravel,boulders)
    

    
ent_file = r"C:\workspace\GLCM\d_5_and_angle_0\entropy_3_zonal_stats_merged.csv"
homo_file = r"C:\workspace\GLCM\d_5_and_angle_0\homo_3_zonal_stats_merged.csv"
var_file = r"C:\workspace\GLCM\d_5_and_angle_0\var_3_zonal_stats_merged.csv"



ent_df = pd.read_csv(ent_file, sep=',')
homo_df = pd.read_csv(homo_file,sep=',')
var_df = pd.read_csv(var_file,sep=',')


ent_cv = variation(ent_df['percentile_50'].dropna().values)
homo_cv = variation(homo_df['percentile_50'].dropna().values)
var_cv = variation(var_df['percentile_50'].dropna().values)

print ent_cv, homo_cv, var_cv
#STD_weights
ent_weight = find_weights(ent_df)
homo_weight = find_weights(homo_df)
var_weight = find_weights(var_df)
print ent_weight, homo_weight, var_weight


print ent_cv/(ent_cv+ homo_cv+ var_cv), homo_cv/(ent_cv+ homo_cv+ var_cv), var_cv/(ent_cv+ homo_cv+ var_cv)