import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from pandas.tools.plotting import table

def centeroidnp(df,df1,df2,query1,metric):
    length = df.query(query1)[metric].dropna().values.size
    sum_x = np.nansum(df.query(query1)[metric].values)
    sum_y = np.nansum(df1.query(query1)[metric].values)
    sum_z = np.nansum(df2.query(query1)[metric].values)
    return sum_x/length, sum_y/length, sum_z/length
    
#    x = np.nanmedian(df.query(query1)[metric])
#    y = np.nanmedian(df1.query(query1)[metric])
#    z= np.nanmedian(df2.query(query1)[metric])
#    return x,y,z
    
def error_bars(df,df1,df2,query1,metric):
    y_err = np.nanstd(df.query(query1)[metric].values)
    x_err = np.nanstd(df1.query(query1)[metric].values)
    z_err = np.nanstd(df1.query(query1)[metric].values)
    return x_err, y_err, z_err
    
    
homo = r"C:\workspace\GLCM\d_5_and_angle_0\homo_3_zonal_stats_merged.csv"
var = r"C:\workspace\GLCM\d_5_and_angle_0\var_3_zonal_stats_merged.csv"
ent = r"C:\workspace\GLCM\d_5_and_angle_0\entropy_3_zonal_stats_merged.csv"

homo_df = pd.read_csv(homo,sep=',',usecols=['percentile_25','percentile_50','percentile_75','substrate'])
var_df = pd.read_csv(var, sep=',',usecols=['percentile_25','percentile_50','percentile_75','substrate'])
ent_df = pd.read_csv(ent, sep=',',usecols=['percentile_25','percentile_50','percentile_75','substrate'])


s_query = 'substrate == "sand"'
g_query = 'substrate == "gravel"'
b_query = 'substrate == "boulders"'

#Centroid Calculation
s_centroid = centeroidnp(homo_df,var_df,ent_df,s_query,'percentile_25')    
g_centroid = centeroidnp(homo_df,var_df,ent_df,g_query,'percentile_25')    
b_centroid = centeroidnp(homo_df,var_df,ent_df,b_query,'percentile_25')    
s_err = error_bars(homo_df,var_df,ent_df,s_query,'percentile_25')    
g_err = error_bars(homo_df,var_df,ent_df,g_query,'percentile_25')
b_err = error_bars(homo_df,var_df,ent_df,b_query,'percentile_25')

cent_df = pd.DataFrame(columns=['x_cent','y_cent','z_cent','y_err','x_err','z_err'], index=['sand','gravel','boulders'])
cent_df.loc['sand'] = pd.Series({'x_cent':s_centroid[0] ,'y_cent':s_centroid[1] ,'z_cent':s_centroid[2],'y_err':s_err[1] ,'x_err':s_err[0],'z_err':s_err[2]})
cent_df.loc['gravel'] = pd.Series({'x_cent':g_centroid[0] ,'y_cent': g_centroid[1],'z_cent':g_centroid[2],'y_err':g_err[1] ,'x_err':g_err[0],'z_err':g_err[2]})
cent_df.loc['boulders'] = pd.Series({'x_cent':b_centroid[0] ,'y_cent':b_centroid[1] ,'z_cent':b_centroid[2],'y_err': b_err[1],'x_err':b_err[0],'z_err':b_err[2]})
cent_df = cent_df.reset_index()


fig = plt.figure(figsize=(7,10))
ax = fig.add_subplot(311, projection='3d')

ax.scatter(homo_df.query(s_query)['percentile_25'],var_df.query(s_query)['percentile_25'],ent_df.query(s_query)['percentile_25'],c='blue')
ax.scatter(s_centroid[0],s_centroid[1],s_centroid[2], s = 250, c='blue')
ax.scatter(homo_df.query(g_query)['percentile_25'],var_df.query(g_query)['percentile_25'],ent_df.query(g_query)['percentile_25'],c='green')
ax.scatter(g_centroid[0],g_centroid[1],g_centroid[2], s = 250, c='green')
ax.scatter(homo_df.query(b_query)['percentile_25'],var_df.query(b_query)['percentile_25'],ent_df.query(b_query)['percentile_25'],c='red')
ax.scatter(b_centroid[0],b_centroid[1],b_centroid[2], s = 250, c='red')
#the_table = table(ax, cent_df.set_index('index').applymap(lambda x: round(x,3)),bbox_to_anchor=(1.05, 1), loc=2,colWidths=[0.1,0.1,0.1,0.1,0.1,0.1],zorder=0)
ax.set_xlabel('homo_25')
ax.set_ylabel('var_25')
ax.set_zlabel('ent_25')

ax = fig.add_subplot(312, projection='3d')
#Centroid Calculation
s_centroid = centeroidnp(homo_df,var_df,ent_df,s_query,'percentile_50')    
g_centroid = centeroidnp(homo_df,var_df,ent_df,g_query,'percentile_50')    
b_centroid = centeroidnp(homo_df,var_df,ent_df,b_query,'percentile_50')    
s_err = error_bars(homo_df,var_df,ent_df,s_query,'percentile_50')    
g_err = error_bars(homo_df,var_df,ent_df,g_query,'percentile_50')
b_err = error_bars(homo_df,var_df,ent_df,b_query,'percentile_50')

cent_df = pd.DataFrame(columns=['x_cent','y_cent','z_cent','y_err','x_err','z_err'], index=['sand','gravel','boulders'])
cent_df.loc['sand'] = pd.Series({'x_cent':s_centroid[0] ,'y_cent':s_centroid[1] ,'z_cent':s_centroid[2],'y_err':s_err[1] ,'x_err':s_err[0],'z_err':s_err[2]})
cent_df.loc['gravel'] = pd.Series({'x_cent':g_centroid[0] ,'y_cent': g_centroid[1],'z_cent':g_centroid[2],'y_err':g_err[1] ,'x_err':g_err[0],'z_err':g_err[2]})
cent_df.loc['boulders'] = pd.Series({'x_cent':b_centroid[0] ,'y_cent':b_centroid[1] ,'z_cent':b_centroid[2],'y_err': b_err[1],'x_err':b_err[0],'z_err':b_err[2]})
cent_df = cent_df.reset_index()

ax.scatter(homo_df.query(s_query)['percentile_50'],var_df.query(s_query)['percentile_50'],ent_df.query(s_query)['percentile_50'],c='blue')
ax.scatter(s_centroid[0],s_centroid[1],s_centroid[2], s = 250, c='blue')
ax.scatter(homo_df.query(g_query)['percentile_50'],var_df.query(g_query)['percentile_50'],ent_df.query(g_query)['percentile_50'],c='green')
ax.scatter(g_centroid[0],g_centroid[1],g_centroid[2], s = 250, c='green')
ax.scatter(homo_df.query(b_query)['percentile_50'],var_df.query(b_query)['percentile_50'],ent_df.query(b_query)['percentile_50'],c='red')
ax.scatter(b_centroid[0],b_centroid[1],b_centroid[2], s = 250, c='red')

ax.set_xlabel('homo_percentile_50')
ax.set_ylabel('var_percentile_50')
ax.set_zlabel('ent_percentile_50')

ax = fig.add_subplot(313, projection='3d')
#Centroid Calculation
s_centroid = centeroidnp(homo_df,var_df,ent_df,s_query,'percentile_75')    
g_centroid = centeroidnp(homo_df,var_df,ent_df,g_query,'percentile_75')    
b_centroid = centeroidnp(homo_df,var_df,ent_df,b_query,'percentile_75')    
s_err = error_bars(homo_df,var_df,ent_df,s_query,'percentile_75')    
g_err = error_bars(homo_df,var_df,ent_df,g_query,'percentile_75')
b_err = error_bars(homo_df,var_df,ent_df,b_query,'percentile_75')


ax.scatter(homo_df.query(s_query)['percentile_75'],var_df.query(s_query)['percentile_75'],ent_df.query(s_query)['percentile_75'],c='blue')
ax.scatter(s_centroid[0],s_centroid[1],s_centroid[2], s = 250, c='blue')
ax.scatter(homo_df.query(g_query)['percentile_75'],var_df.query(g_query)['percentile_75'],ent_df.query(g_query)['percentile_75'],c='green')
ax.scatter(g_centroid[0],g_centroid[1],g_centroid[2], s = 250, c='green')
ax.scatter(homo_df.query(b_query)['percentile_75'],var_df.query(b_query)['percentile_75'],ent_df.query(b_query)['percentile_75'],c='red')
ax.scatter(b_centroid[0],b_centroid[1],b_centroid[2], s = 250, c='red')

ax.set_xlabel('homo_75')
ax.set_ylabel('var_75')
ax.set_zlabel('ent_75')

plt.suptitle('3 Meter Grid: GLCM Properties')
plt.tight_layout()
plt.savefig(r"C:\workspace\GLCM\d_5_and_angle_0\ent_var_homo_3d.png", dpi=600)


