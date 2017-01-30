# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 09:34:47 2017

@author: dan
"""
from __future__ import division
import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from sklearn import mixture
from sklearn import cross_validation
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
import pandas as pd
# =========================================================
def remove_outliers(X, y, k):
   """
   simple outlier removal based on deviation from mean
   """
   mu, sigma = np.mean(X, axis=0), np.std(X, axis=0, ddof=1)
   index = np.all(np.abs((X - mu) / sigma) < k, axis=1)
   return X[index], y[index]

#===========================================================================

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
      g.means_init =  np.array([X_train[y_train == i].mean(axis=0) for i in range(1,len(classes)+1)])
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
      y_test_pred = g.predict(np.expand_dims(X_test, axis=1))+1
      test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
      print "======================================="
      print "test scores: "+predictors[n]
      print test_accuracy

      print(classification_report(y_test_pred.ravel(), y_test.ravel()))

      # show normalized confusion matrix
      cm = confusion_matrix(y_test.ravel(), y_test_pred.ravel())
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print(cm)
      
      # plot
      D = X_train.ravel()
      xmin = D.min()
      xmax = D.max()
      x = np.linspace(xmin,xmax,1000)

      d, bins = np.histogram(D, 100)
      
      plt.subplot()
      
      plt.bar(bins[1:], d/np.max(d), color='k', alpha=0.5, width=0.05)
      col='rg'
      for k in range(len(classes)):
         mu = g.means_[k]
         try:
            sigma = np.sqrt(g.covariances_[k])
         except:
            sigma = np.sqrt(g.covariances_)
         yy  = np.squeeze(g.weights_[k]*stats.norm.pdf(x,mu,sigma))
         plt.plot(x,yy/np.max(yy), c=col[k], linewidth=1, label=classes[k])
         
      plt.xlabel('Homogeneity')
      plt.legend(fontsize=6)
      plt.setp(plt.xticks()[1], fontsize=7)#, rotation=45)
      plt.setp(plt.yticks()[1], fontsize=7)#, rotation=45)
      plt.title('Non-Bayesian, Accuracy: '+str(test_accuracy)[:4] +'\n Covariance Type: ' + covtype + ' BIC = ' + str(int(bic)))
      
      
      #============================================================================
      grouped = tmp_df.groupby('sedclass')
      sand = grouped.get_group(1)
      boulders = grouped.get_group(3)
      
      fig,ax = plt.subplots()
      sand['Entropy'].plot.hist(bins=50,color=['r'],ax=ax,alpha=0.6,normed=True,label='Sand')
      boulders['Entropy'].plot.hist(bins=50,color=['g'],ax=ax,alpha=0.6,normed=True,label='Boulders')
      #ax.bar(bins[1:], d/np.max(d), color='k', alpha=0.5, width=0.05)
      ax2 = ax.twinx()
      col='rg'
      for k in range(len(classes)):
         mu = g.means_[k]
         try:
            sigma = np.sqrt(g.covariances_[k])
         except:
            sigma = np.sqrt(g.covariances_)
         yy  = np.squeeze(g.weights_[k]*stats.norm.pdf(x,mu,sigma))
         ax2.plot(x,yy/np.max(yy), c=col[k], linewidth=1, label=classes[k])
      ax2.set_yticklabels('')
      ax.set_xlabel('Entropy')
      ax.legend(fontsize=10)
      plt.setp(plt.xticks()[1], fontsize=7)#, rotation=45)
      plt.setp(plt.yticks()[1], fontsize=7)#, rotation=45)
      plt.title('Non-Bayesian, Accuracy: '+str(test_accuracy)[:4] +'\n Covariance Type: ' + covtype + ' BIC = ' + str(int(bic)))
      
      
      
      
      
      
      
###################################################################################################################
#####################Variance, 3 goussicans, 3 sed classes
###################################################################################################################
csv = r"C:\workspace\GLCM\new_output\merged_aggregraded_distributions.csv"
data = np.genfromtxt(csv, delimiter=',', skip_header=1)

data = data[~np.isnan(data).any(axis=1)]
# for removing outliers
factor=3 #smaller the number, the more ruthless the pruning

data = data
data, sedclass = remove_outliers(data[:,:3], data[:,3], factor)

tmp_df = pd.DataFrame({'Variance':data[:,1],'sedclass':sedclass})


data[:,2] = 1 - data[:,2]
predictors = ['Entropy','Variance','Homogeneity']



classes = ['Sand','Gravel','Boulders']

standardize = 1

for covtype in ['full']:
   print 'Working on covariance type %s...' %(covtype,)
   
   for n in [1]:
      print "working on "+predictors[n]
      print "Working on GMM..."
      # split into 50% training, 50% testing
      if standardize==1: # standardize data
         X_train, X_test, y_train, y_test = cross_validation.train_test_split(preprocessing.scale(tmp_df['Variance'].values), tmp_df['sedclass'].values, test_size=0.5, random_state=0)
         tmp_df['Variance'] =preprocessing.scale(tmp_df['Variance'].values)
      else:
         X_train, X_test, y_train, y_test = cross_validation.train_test_split(tmp_df['Variance'].values, tmp_df['sedclass'].values, test_size=0.5, random_state=0)
    
      #initialize the GMM with means
      g = mixture.GaussianMixture(n_components=3, max_iter=100, random_state=0, covariance_type=covtype)
      g.means_init =  np.array([X_train[y_train == i].mean(axis=0) for i in range(1,len(classes)+1)])
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
      y_test_pred = g.predict(np.expand_dims(X_test, axis=1))+1
      test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
      print "======================================="
      print "test scores: "+predictors[n]
      print test_accuracy

      print(classification_report(y_test_pred.ravel(), y_test.ravel()))

      # show normalized confusion matrix
      cm = confusion_matrix(y_test.ravel(), y_test_pred.ravel())
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print(cm)
      
      # plot
      D = X_train.ravel()
      xmin = D.min()
      xmax = D.max()
      x = np.linspace(xmin,xmax,1000)

      d, bins = np.histogram(D, 100)
      
      plt.subplot()
      
      plt.bar(bins[1:], d/np.max(d), color='k', alpha=0.5, width=0.05)
      col='rgb'
      for k in range(len(classes)):
         mu = g.means_[k]
         try:
            sigma = np.sqrt(g.covariances_[k])
         except:
            sigma = np.sqrt(g.covariances_)
         yy  = np.squeeze(g.weights_[k]*stats.norm.pdf(x,mu,sigma))
         plt.plot(x,yy/np.max(yy), c=col[k], linewidth=1, label=classes[k])
         
      plt.xlabel('Variance')
      plt.legend(fontsize=6)
      plt.setp(plt.xticks()[1], fontsize=7)#, rotation=45)
      plt.setp(plt.yticks()[1], fontsize=7)#, rotation=45)
      plt.title('Non-Bayesian, Accuracy: '+str(test_accuracy)[:4] +'\n Covariance Type: ' + covtype + ' BIC = ' + str(int(bic)))
      
      
      #============================================================================
      grouped = tmp_df.groupby('sedclass')
      sand = grouped.get_group(1)
      gravel = grouped.get_group(2)
      boulders = grouped.get_group(3)
      
      fig,ax = plt.subplots()
      sand['Variance'].plot.hist(bins=50,color=['r'],ax=ax,alpha=0.6,normed=True,label='Sand')
      gravel['Variance'].plot.hist(bins=50,color=['g'],ax=ax,alpha=0.6,normed=True,label='Gravel')
      boulders['Variance'].plot.hist(bins=50,color=['b'],ax=ax,alpha=0.6,normed=True,label='Boulders')
      #ax.bar(bins[1:], d/np.max(d), color='k', alpha=0.5, width=0.05)
      ax2 = ax.twinx()
     
      for k in range(len(classes)):
         mu = g.means_[k]
         try:
            sigma = np.sqrt(g.covariances_[k])
         except:
            sigma = np.sqrt(g.covariances_)
         yy  = np.squeeze(g.weights_[k]*stats.norm.pdf(x,mu,sigma))
         ax2.plot(x,yy/np.max(yy), c=col[k], linewidth=1, label=classes[k])
      ax2.set_yticklabels('')
      ax.set_xlabel('Variance')
      ax.legend(fontsize=10)
      plt.setp(plt.xticks()[1], fontsize=7)#, rotation=45)
      plt.setp(plt.yticks()[1], fontsize=7)#, rotation=45)
      plt.title('Non-Bayesian, Accuracy: '+str(test_accuracy)[:4] +'\n Covariance Type: ' + covtype + ', BIC = ' + str(int(bic)))

