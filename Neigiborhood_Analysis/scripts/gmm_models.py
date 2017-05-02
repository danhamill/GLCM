# -*- coding: utf-8 -*-
"""
Created on Tue Feb 07 14:41:12 2017

@author: dan
"""
from __future__ import division
import numpy as np
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools
import pytablewriter
from rasterstats import zonal_stats
from scipy import linalg
from sklearn import mixture
from sklearn import cross_validation
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from sklearn import preprocessing
import pandas as pd
import os
from osgeo import gdal,ogr,osr


def read_raster(raster):
    ds = gdal.Open(raster)
    data = ds.GetRasterBand(1).ReadAsArray()
    gt = ds.GetGeoTransform()
    return data, gt

# =========================================================
def get_subs(shp):
    ds = ogr.Open(shp)
    lyr = ds.GetLayer(0)
    a=[]
    for row in lyr:
        a.append(row.substrate)
    lyr.ResetReading()
    del ds
    return a

# =========================================================
def remove_outliers(X, y, k):
   """
   simple outlier removal based on deviation from mean
   """
   mu, sigma = np.mean(X, axis=0), np.std(X, axis=0, ddof=1)
   index = np.all(np.abs((X - mu) / sigma) < k, axis=1)
   return X[index], y[index]

# =========================================================
def CreateRaster(sed_class,gt,outFile):  
    '''
    Exports data to GTiff Raster
    '''
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(26949)
    sed_class = np.squeeze(sed_class)
    sed_class[np.isnan(sed_class)] = -99
    driver = gdal.GetDriverByName('GTiff')
    rows,cols = np.shape(sed_class)
    ds = driver.Create( outFile, cols, rows, 1, gdal.GDT_Float32)      
    if proj is not None:  
        ds.SetProjection(proj.ExportToWkt()) 
    ds.SetGeoTransform(gt)
    ss_band = ds.GetRasterBand(1)
    ss_band.WriteArray(sed_class)
    ss_band.SetNoDataValue(-99)
    ss_band.FlushCache()
    ss_band.ComputeStatistics(False)
    del ds   
   
###################################################################################################################
#####################Entropy, 2 goussicans, 2 sed classes
###################################################################################################################
csv = r"C:\workspace\GLCM\new_output\merged_aggregraded_distributions.csv"
data = np.genfromtxt(csv, delimiter=',', skip_header=1)

data = data[~np.isnan(data).any(axis=1)]
# for removing outliers
factor=3 #smaller the number, the more ruthless the pruning

data = data
data, sedclass = remove_outliers(data[:,:3], data[:,3], factor)

tmp_df = pd.DataFrame({'Entropy':data[:,0],'sedclass':sedclass})
tmp_df = tmp_df[tmp_df['sedclass'] != 2]

data[:,2] = 1 - data[:,2]
predictors = ['Entropy','Variance','Homogeneity']



classes = ['Sand','Boulders']

standardize = 0

for covtype in ['tied']:
   print 'Working on covariance type %s...' %(covtype,)
   
   for n in [0]:
      print "working on "+predictors[n]
      print "Working on GMM..."
      # split into 50% training, 50% testing
      if standardize==1: # standardize data
         X_train, X_test, y_train, y_test = cross_validation.train_test_split(preprocessing.scale(tmp_df['Entropy'].values), tmp_df['sedclass'].values, test_size=0.5, random_state=0)
         tmp_df['Entropy'] =preprocessing.scale(tmp_df['Entropy'].values)
      else:
         X_train, X_test, y_train, y_test = cross_validation.train_test_split(tmp_df['Entropy'].values, tmp_df['sedclass'].values, test_size=0.5, random_state=0)
    
      #initialize the GMM with means
      g = mixture.GaussianMixture(n_components=2, max_iter=100, random_state=0, covariance_type=covtype)
      g.means_init =  np.array([X_train[y_train == i].mean(axis=0) for i in [1,3]])
      g.means_init =np.expand_dims(g.means_init, axis=1) 

      # fit the model
      g.fit(np.expand_dims(X_train, axis=1) )
      
   
      #make sure the means are in order
      order = np.argsort(np.squeeze(g.means_))
      g.means_ = g.means_[order]
      try:
         g.covariances_ = g.covariances_[order]
      except:
         pass
      g.weights_ = g.weights_[order]

      
      bic = g.bic(np.expand_dims(X_train, axis=1) )
      # test
      y_test_pred = g.predict(np.expand_dims(X_test, axis=1))
      y_test_pred[y_test_pred==1] = 3
      y_test_pred[y_test_pred==0] = 1
      test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
      print "======================================="
      print "test scores: "+predictors[n]
      print test_accuracy

      print(classification_report(y_test_pred.ravel(), y_test.ravel()))
      print (cohen_kappa_score(y_test_pred.ravel(), y_test.ravel()))

      # show normalized confusion matrix
      cm = confusion_matrix(y_test.ravel(), y_test_pred.ravel())
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print(cm)
         

      
#      # plot
#      D = X_train.ravel()
#      xmin = D.min()
#      xmax = D.max()
#      x = np.linspace(xmin,xmax,1000)
#
#      d, bins = np.histogram(D, 100)
#      
#      plt.subplot()
#      
#      plt.bar(bins[1:], d/np.max(d), color='k', alpha=0.5, width=0.05)
#      col='rg'
#      for k in range(len(classes)):
#         mu = g.means_[k]
#         try:
#            sigma = np.sqrt(g.covariances_[k])
#         except:
#            sigma = np.sqrt(g.covariances_)
#         yy  = np.squeeze(g.weights_[k]*stats.norm.pdf(x,mu,sigma))
#         plt.plot(x,yy/np.max(yy), c=col[k], linewidth=1, label=classes[k])
#         
#      plt.xlabel('Homogeneity')
#      plt.legend(fontsize=6)
#      plt.setp(plt.xticks()[1], fontsize=7)#, rotation=45)
#      plt.setp(plt.yticks()[1], fontsize=7)#, rotation=45)
#      plt.title('Non-Bayesian, Accuracy: '+str(test_accuracy)[:4] +'\n Covariance Type: ' + covtype + ' BIC = ' + str(int(bic)))
#      
#      
#      #============================================================================
#      grouped = tmp_df.groupby('sedclass')
#      sand = grouped.get_group(1)
#      boulders = grouped.get_group(3)
#      
#      fig,ax = plt.subplots()
#      sand['Entropy'].plot.hist(bins=50,color=['r'],ax=ax,alpha=0.6,normed=True,label='Sand')
#      boulders['Entropy'].plot.hist(bins=50,color=['g'],ax=ax,alpha=0.6,normed=True,label='Boulders')
#      #ax.bar(bins[1:], d/np.max(d), color='k', alpha=0.5, width=0.05)
#      ax2 = ax.twinx()
#      col='rg'
#      for k in range(len(classes)):
#         mu = g.means_[k]
#         try:
#            sigma = np.sqrt(g.covariances_[k])
#         except:
#            sigma = np.sqrt(g.covariances_)
#         yy  = np.squeeze(g.weights_[k]*stats.norm.pdf(x,mu,sigma))
#         ax2.plot(x,yy/np.max(yy), c=col[k], linewidth=1, label=classes[k])
#      ax2.set_yticklabels('')
#      ax.set_xlabel('Entropy')
#      ax.legend(fontsize=10)
#      plt.setp(plt.xticks()[1], fontsize=7)#, rotation=45)
#      plt.setp(plt.yticks()[1], fontsize=7)#, rotation=45)
#      plt.title('Non-Bayesian, Accuracy: '+str(test_accuracy)[:4] +'\n Covariance Type: ' + covtype + ' BIC = ' + str(int(bic)))
#      plt.savefig(r'C:\workspace\Research\hamill_thesis\chapter2\figures\Entropy_Model_2_Part.pdf',dpi=600)
      
      # Now lets make some maps...
      ent_dict = {'R01346':r"C:\workspace\GLCM\Neigiborhood_Analysis\GLCM_Rasters\R02028_3_entropy_Resampled.tif"}      
      rasters = []
      for (key,v), in zip(ent_dict.items()):         
          ent_raster =v

          
          ent_data, gt = read_raster(ent_raster)

          
          vec1 = np.c_[ent_data.flatten()]
          vec1[np.isnan(vec1)] = 0; vec1[np.isinf(vec1)] = 0
               
          ind = np.nonzero(vec1)[0]  
          sed_class = np.zeros(np.shape(ent_data.flatten()))*np.nan
          
          for k in xrange(len(ind)):
              sed_class[ind[k]] = g.predict(vec1[ind[k]].reshape(1, -1))
          sed_class[sed_class == 1] = 3
          sed_class[sed_class == 0] = 1
      
          sed_class = np.reshape(sed_class,np.shape(read_raster(ent_raster)[0]))
          
          outFile = r'C:\workspace\GLCM\Neigiborhood_Analysis\Sedclass_Rasters' + os.sep + str(key) + '_GMM_2class_raster.tif'
          rasters.append(outFile)
          CreateRaster(sed_class,gt,outFile)
          
#      for (key,v), in zip(ent_dict.items()):         
#          ent_raster =v
#
#          
#          ent_data, gt = read_raster(ent_raster)
#
#          
#          vec1 = np.c_[ent_data.flatten()]
#          vec1[np.isnan(vec1)] = 0; vec1[np.isinf(vec1)] = 0
#               
#          ind = np.nonzero(vec1)[0]  
#          sed_class = np.zeros(np.shape(ent_data.flatten()))*np.nan
#          
#          for k in xrange(len(ind)):
#               sed_class[ind[k]] = g.predict_proba(vec1[ind[k]].reshape(1, -1))[:,g.predict(vec1[ind[k]].reshape(1, -1))[0]]
#               
##          sed_class[sed_class == 1] = 2
##          sed_class[sed_class == 0] = 1
#      
#          sed_class = np.reshape(sed_class,np.shape(read_raster(ent_raster)[0]))
#          
#          outFile = r'C:\workspace\GLCM\Neigiborhood_Analysis\Sedclass_Rasters' + os.sep + str(key) + '_GMM2class_proba_raster.tif'
#          CreateRaster(sed_class,gt,outFile)

##Classification Results
#shps = [r"C:\workspace\Merged_SS\window_analysis\shapefiles\tex_seg_2014_09_67_3class.shp",
#        r"C:\workspace\Merged_SS\window_analysis\shapefiles\tex_seg_800_3class.shp", 
#          r"C:\workspace\Merged_SS\window_analysis\shapefiles\R01765.shp"]
#for n in xrange(len(rasters)):
#    raster = rasters[n]
#    shp = shps[n]
#    subs = get_subs(shp)
#    stats = zonal_stats(shp, raster, categorical=True, nodata=-99)
#    if raster == rasters[0]:
#        merge_subs = subs
#        merge_df = pd.DataFrame(stats)
#    else:
#        merge_subs.extend(subs)
#        merge_df = pd.concat([merge_df,pd.DataFrame(stats)])
#        del stats, shp,raster,subs
#
#del n
#
#merge_df['substrate'] = merge_subs
#
#merge_df.rename(columns = {1.0:'Sand',3.0:'Boulders'},inplace=True)
#merge_df = merge_df[['Sand','Boulders','substrate']]
#merge_df = merge_df[merge_df['substrate'] != 'gravel']
#
#pvt = pd.pivot_table(merge_df, index=['substrate'],values=['Sand','Boulders'],aggfunc=np.nansum)
#del merge_df
#  #Percentage classification table
#class_df = pvt.div(pvt.sum(axis=1), axis=0)
#
#writer = pytablewriter.MarkdownTableWriter()
#writer.table_name = "Variance and Homogeneity GMM Model Results"
#writer.header_list = list(class_df.columns.values)
#writer.value_matrix = class_df.values.tolist()
#writer.write_table()   
      
###################################################################################################################
#####################Variance + Homogeneity,4 goussicans
###################################################################################################################
csv = r"C:\workspace\GLCM\new_output\merged_aggregraded_distributions.csv"
data = np.genfromtxt(csv, delimiter=',', skip_header=1)

data = data[~np.isnan(data).any(axis=1)]
# for removing outliers
factor=3 #smaller the number, the more ruthless the pruning

data = data
data, sedclass = remove_outliers(data[:,:3], data[:,3], factor)
data[:,2] = 1 - data[:,2]
tmp_df = pd.DataFrame({'Variance':data[:,1],'Homogeneity':data[:,2],'sedclass':sedclass})



predictors = ['Entropy','Variance','Homogeneity']



classes = ['Sand','Gravel','Boulders']

standardize = 0

for covtype in ['full']:
   print 'Working on covariance type %s...' %(covtype,)
   
   for n in [1]:
      print "working on "+predictors[n]
      print "Working on GMM..."
      # split into 50% training, 50% testing
      if standardize==1: # standardize data
         X_train, X_test, y_train, y_test = cross_validation.train_test_split(preprocessing.robust_scale(tmp_df[['Variance','Homogeneity']].values), tmp_df['sedclass'].values, test_size=0.5, random_state=0)
         tmp_df['Variance'] = preprocessing.robust_scale(tmp_df[['Variance']].values)
         tmp_df['Homogeneity'] = preprocessing.robust_scale(tmp_df[['Variance']].values)        
      else:
         X_train, X_test, y_train, y_test = cross_validation.train_test_split(tmp_df[['Variance','Homogeneity']].values, tmp_df['sedclass'].values, test_size=0.5, random_state=0)

         
      #initialize the GMM with means
      g = mixture.GaussianMixture(n_components=4, max_iter=100, random_state=0, covariance_type=covtype)
      
      means =  np.array([X_train[y_train == i].mean(axis=0) for i in range(1,len(classes)+1)])
      
      df = pd.DataFrame(data=means,columns=['Variance','Homogeneity'])
      
      
      df2 = pd.DataFrame(index=['sand','sand_gravel','gravel_boluders','boulders'],columns = ['Variance','Homogeneity'])
      df2.loc['sand'] = pd.Series({'Variance':df.iloc[int(0)].Variance,
                                    'Homogeneity':df.iloc[int(0)].Homogeneity})
      df2.loc['sand_gravel'] = pd.Series({'Variance': (df.iloc[1].Variance+ df.iloc[0].Variance)/2,'Homogeneity': (df.iloc[1].Homogeneity+ df.iloc[0].Homogeneity)/2})
      df2.loc['gravel_boluders'] = pd.Series({'Variance': (df.iloc[2].Variance+ df.iloc[1].Variance)/2,'Homogeneity': 
                                          (df.iloc[2].Homogeneity+ df.iloc[1].Homogeneity)/2})
      df2.loc['boulders'] = pd.Series({'Variance':df.iloc[2].Variance,
                                      'Homogeneity':df.iloc[2].Homogeneity})
      means = df2.values
      del df, df2
      means = means.astype('float')
      
      #g.means_init =  np.array([X_train[y_train == i].mean(axis=0) for i in range(1,len(classes)+1)])
      g.means_init = means
      # fit the model
      g.fit(X_train )
   
      #make sure the means are in order
      order = np.argsort(g.means_[:,0])
      g.means_[:,0] = g.means_[:,0][order]
      g.covariances_[:,0] = g.covariances_[:,0][order]
        
      bic = g.bic(X_train)

      # test
      y_test_pred = g.predict(X_test)
      y_test_pred[y_test_pred==1]=2
      y_test_pred[y_test_pred==0]=1
      test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
      print "======================================="
      print "test scores: "+predictors[n]
      print test_accuracy

      print(classification_report(y_test_pred.ravel(), y_test.ravel()))
      print (cohen_kappa_score(y_test_pred.ravel(), y_test.ravel()))
      # show normalized confusion matrix
      cm = confusion_matrix(y_test.ravel(), y_test_pred.ravel())
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print(cm)
      
      
      
#      
#      color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue','darkorange'])
#    
#     # Plot the winner
#      splot = plt.subplot(1, 1, 1)
#      Y_ = g.predict(X_train)
#      #Y_[Y_==1]=2
#      #Y_[Y_==0]=1      
#      labels = ['Sand','Sand_Gravel','Gravel_Boulders','Boulders']
#      n = 0
#      for i, (mean, cov, color) in enumerate(zip(g.means_, g.covariances_,
#                                               color_iter)):
#          v, w = linalg.eigh(cov)
#          if not np.any(Y_ == i):
#              continue
#          plt.scatter(X_train[Y_ == i, 0], X_train[Y_ == i, 1], .8, color=color,label= labels[n])
#    
#          # Plot an ellipse to show the Gaussian component
#          angle = np.arctan2(w[0][1], w[0][0])
#          angle = 180. * angle / np.pi  # convert to degrees
#          v = 2. * np.sqrt(2.) * np.sqrt(v)
#          ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
#          ell.set_clip_box(splot.bbox)
#          ell.set_alpha(.5)
#          splot.add_artist(ell)
#          n+=1
#          
#      y_train_pred = g.predict(X_train) 
#      y_train_pred[y_train_pred==1]=2
#      y_train_pred[y_train_pred==0]=1
#      train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
#      plt.text(0.05, 0.9, 'Train accuracy: %.1f' % train_accuracy,
#              transform=splot.transAxes)
#
#      y_test_pred = g.predict(X_test)
#      y_test_pred[y_test_pred==1]=2
#      y_test_pred[y_test_pred==0]=1
#      test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
#      plt.text(0.05, 0.8, 'Test accuracy: %.1f' % test_accuracy,
#              transform=splot.transAxes)
#      plt.legend(fontsize='medium')
#      plt.xticks(())
#      plt.yticks(())
#      plt.title('Variance+Homogeneity: full covariance model \n 4 fitted gaussians ' + ' BIC = ' + str(int(bic)))
#      plt.subplots_adjust(hspace=.35, bottom=.02)
#      plt.show()
#      plt.savefig(r'C:\workspace\Research\hamill_thesis_2\chapter2\figures\var_homo_mixture.pdf',dpi=600)
      
      
      # Now lets make some maps...
      var_dict = {'R02028':r"C:\workspace\GLCM\Neigiborhood_Analysis\GLCM_Rasters\R02028_3_var_Resampled.tif"}       
    
      homo_dict = {'R02028':r"C:\workspace\GLCM\Neigiborhood_Analysis\GLCM_Rasters\R02028_3_homo_Resampled.tif"}  
                

      rasters = []
      for (key,v), (key1,v1)in zip(var_dict.items(), homo_dict.items())[0:1]:         
          var_raster =v
          homo_raster =v1
          
          var_data, gt = read_raster(var_raster)
          homo_data = read_raster(homo_raster)[0]
          
          vec1 = np.c_[var_data.flatten(),1-homo_data.flatten()]
          vec1[np.isnan(vec1)] = 0; vec1[np.isinf(vec1)] = 0
               
          ind = np.nonzero(vec1)[0]  
          sed_class = np.zeros(np.shape(var_data.flatten()))*np.nan
          
          for k in xrange(len(ind)):
              sed_class[ind[k]] = g.predict(vec1[ind[k]].reshape(1, -1))
          sed_class[sed_class == 1] = 2
          sed_class[sed_class == 0] = 1
      
          sed_class = np.reshape(sed_class,np.shape(read_raster(homo_raster)[0]))
          
          outFile = r'C:\workspace\GLCM\Neigiborhood_Analysis\Sedclass_Rasters' + os.sep + str(key) + '_GMM_3class_raster.tif'
          rasters.append(outFile)
          CreateRaster(sed_class,gt,outFile)
     
#      for (key,v), (key1,v1)in zip(var_dict.items(), homo_dict.items())[0:1]:         
#          var_raster =v
#          homo_raster =v1
#          
#          var_data, gt = read_raster(var_raster)
#          homo_data = read_raster(homo_raster)[0]
#          
#          vec1 = np.c_[var_data.flatten(),1-homo_data.flatten()]
#          vec1[np.isnan(vec1)] = 0; vec1[np.isinf(vec1)] = 0
#               
#          ind = np.nonzero(vec1)[0]  
#          sed_class = np.zeros(np.shape(var_data.flatten()))*np.nan
#          
#          for k in xrange(len(ind)):
#               sed_class[ind[k]] = g.predict_proba(vec1[ind[k]].reshape(1, -1))[:,g.predict(vec1[ind[k]].reshape(1, -1))[0]]
#               
##          sed_class[sed_class == 1] = 2
##          sed_class[sed_class == 0] = 1
#      
#          sed_class = np.reshape(sed_class,np.shape(read_raster(homo_raster)[0]))
#          
#          outFile = r'C:\workspace\GLCM\new_output\gmm_rasters' + os.sep + str(key) + '_GMM3class_proba_raster.tif'
#          CreateRaster(sed_class,gt,outFile)
#          
#          
#shps = [r"C:\workspace\Merged_SS\window_analysis\shapefiles\tex_seg_2014_09_67_3class.shp",
#        r"C:\workspace\Merged_SS\window_analysis\shapefiles\tex_seg_800_3class.shp", 
#          r"C:\workspace\Merged_SS\window_analysis\shapefiles\R01765.shp"]
#for n in xrange(len(rasters)):
#    raster = rasters[n]
#    shp = shps[n]
#    subs = get_subs(shp)
#    stats = zonal_stats(shp, raster, categorical=True, nodata=-99)
#    if raster == rasters[0]:
#        merge_subs = subs
#        merge_df = pd.DataFrame(stats)
#    else:
#        merge_subs.extend(subs)
#        merge_df = pd.concat([merge_df,pd.DataFrame(stats)])
#        del stats, shp,raster,subs
#
#del n
#
#merge_df['substrate'] = merge_subs
#
#merge_df.rename(columns = {1.0:'Sand',2.0:'Gravel',3.0:'Boulders'},inplace=True)
#merge_df = merge_df[['Sand','Gravel','Boulders','substrate']]
#
#pvt = pd.pivot_table(merge_df, index=['substrate'],values=['Sand','Gravel','Boulders'],aggfunc=np.nansum)
#del merge_df
#  #Percentage classification table
#class_df = pvt.div(pvt.sum(axis=1), axis=0).reset_index()
#
#writer = pytablewriter.MarkdownTableWriter()
#writer.table_name = "Variance and Homogeneity GMM Model Results"
#writer.header_list = list(class_df.columns.values)
#writer.value_matrix = class_df.values.tolist()
#writer.write_table()   