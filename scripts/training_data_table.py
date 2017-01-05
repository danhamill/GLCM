# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 08:05:01 2016

@author: dan
"""

import pandas as pd
from glob import glob
import numpy as np

merge_df = pd.DataFrame(columns=['Window_Size','Contrast','Dissimilarity','ASM','Entropy','Correlation','Homogeneity','Substrate'])
    
glob_str = r"C:\workspace\GLCM\d_5_and_angle_0\*_zonal_stats_merged.csv"
csvs = glob(glob_str)
lu_dict = {'asm':'ASM','contrast':'Contrast','corr':'Correlation','diss':'Dissimilarity','energy':'Energy','homo':'Homogeneity'}

ASM_list = csvs[0:5]
cont_list = csvs[5:10]
corr_list = csvs[10:15]
diss_list = csvs[15:20]
eng_list = csvs[20:25]
ent_list = csvs[25:30]
homo_list = csvs[30:35]

grid_list = [10,10,10,20,20,20,2,2,2,3,3,3,5,5,5]
subs_list = ['Boulders','Gravel','Sand','Boulders','Gravel','Sand','Boulders','Gravel','Sand','Boulders','Gravel','Sand','Boulders','Gravel','Sand']
#ASM list
a = []
for n in xrange(len(ASM_list)):
    csv = csvs[n]
    grid_size = csv.split('\\')[-1].split('_')[1]
    metric = lu_dict[csv.split('\\')[-1].split('_')[0]]
    df = pd.read_csv(csv,sep=',')
    df = pd.pivot_table(df,index=['substrate'],values=['mean'],aggfunc = np.nanmean)
    a.extend(df['mean'].values)
    
con = []
for n in xrange(len(cont_list)):
    csv = csvs[n]
    grid_size = csv.split('\\')[-1].split('_')[1]
    metric = lu_dict[csv.split('\\')[-1].split('_')[0]]
    df = pd.read_csv(csv,sep=',')
    df = pd.pivot_table(df,index=['substrate'],values=['mean'],aggfunc = np.nanmean)
    con.extend(df['mean'].values)
    

cor = []
for n in xrange(len(corr_list)):
    csv = csvs[n]
    grid_size = csv.split('\\')[-1].split('_')[1]
    metric = lu_dict[csv.split('\\')[-1].split('_')[0]]
    df = pd.read_csv(csv,sep=',')
    df = pd.pivot_table(df,index=['substrate'],values=['mean'],aggfunc = np.nanmean)
    cor.extend(df['mean'].values)
    
dis = []
for n in xrange(len(diss_list)):
    csv = csvs[n]
    grid_size = csv.split('\\')[-1].split('_')[1]
    metric = lu_dict[csv.split('\\')[-1].split('_')[0]]
    df = pd.read_csv(csv,sep=',')
    df = pd.pivot_table(df,index=['substrate'],values=['mean'],aggfunc = np.nanmean)
    dis.extend(df['mean'].values)
    
eng = []
for n in xrange(len(eng_list)):
    csv = csvs[n]
    grid_size = csv.split('\\')[-1].split('_')[1]
    metric = lu_dict[csv.split('\\')[-1].split('_')[0]]
    df = pd.read_csv(csv,sep=',')
    df = pd.pivot_table(df,index=['substrate'],values=['mean'],aggfunc = np.nanmean)
    eng.extend(df['mean'].values)
       
ent = []
for n in xrange(len(ent_list)):
    csv = csvs[n]
    grid_size = csv.split('\\')[-1].split('_')[1]
    metric = lu_dict[csv.split('\\')[-1].split('_')[0]]
    df = pd.read_csv(csv,sep=',')
    df = pd.pivot_table(df,index=['substrate'],values=['mean'],aggfunc = np.nanmean)
    ent.extend(df['mean'].values)
    
homo = []
for n in xrange(len(homo_list)):
    csv = csvs[n]
    grid_size = csv.split('\\')[-1].split('_')[1]
    metric = lu_dict[csv.split('\\')[-1].split('_')[0]]
    df = pd.read_csv(csv,sep=',')
    df = pd.pivot_table(df,index=['substrate'],values=['mean'],aggfunc = np.nanmean)
    homo.extend(df['mean'].values)
    
asm_df = pd.DataFrame({'ASM':a, 'Substrate':subs_list,'Window_Size':grid_list}).set_index(['Window_Size','Substrate'])
cont_df = pd.DataFrame({'Contrast':con, 'Substrate':subs_list,'Window_Size':grid_list}).set_index('Window_Size','Substrate')
corr_df = pd.DataFrame({'Correlation':cor, 'Substrate':subs_list,'Window_Size':grid_list}).set_index('Window_Size','Substrate')
dis_df = pd.DataFrame({'Dissimilarity':dis, 'Substrate':subs_list,'Window_Size':grid_list}).set_index('Window_Size','Substrate')
eng_df = pd.DataFrame({'Energy':eng, 'Substrate':subs_list,'Window_Size':grid_list}).set_index('Window_Size','Substrate')
ent_df = pd.DataFrame({'Entropy':ent, 'Substrate':subs_list,'Window_Size':grid_list}).set_index('Window_Size','Substrate')
homo_df = pd.DataFrame({'Homogeneity':homo, 'Substrate':subs_list,'Window_Size':grid_list}).set_index('Window_Size','Substrate')


merge_df = pd.DataFrame({'Window_Size':grid_list,'Contrast':cont_df['Contrast'].values,'Dissimilarity':dis_df['Dissimilarity'].values,'ASM':asm_df['ASM'].values,
                         'Energy':eng_df['Energy'].values,'Entropy':ent_df['Entropy'].values,'Correlation':corr_df['Correlation'].values,'Homogeneity':homo_df['Homogeneity'].values,
                         'Substrate':subs_list})
                         
sand_df =merge_df.query('Substrate == "Sand"').set_index('Window_Size').sort_index()       
gravel_df= merge_df.query('Substrate == "Gravel"').set_index('Window_Size').sort_index()    
boulders_df = merge_df.query('Substrate == "Boulders"').set_index('Window_Size').sort_index()    

print sand_df[['Contrast','Dissimilarity','ASM','Energy','Entropy','Correlation','Homogeneity']].to_latex()
print gravel_df[['Contrast','Dissimilarity','ASM','Energy','Entropy','Correlation','Homogeneity']].to_latex()
print boulders_df[['Contrast','Dissimilarity','ASM','Energy','Entropy','Correlation','Homogeneity']].to_latex()
sand_df.sort_index()




















