# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 09:34:47 2017

@author: dan
"""
from __future__ import division
import numpy as np
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import itertools

from scipy import linalg
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

standardize = 0

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

      
      bic = g.bic(np.expand_dims(X_train, axis=1))
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


      
###################################################################################################################
#####################Homogeniety, 2 goussicans, 2 sed classes
###################################################################################################################
csv = r"C:\workspace\GLCM\new_output\merged_aggregraded_distributions.csv"
data = np.genfromtxt(csv, delimiter=',', skip_header=1)

data = data[~np.isnan(data).any(axis=1)]
# for removing outliers
factor=3 #smaller the number, the more ruthless the pruning

data = data
data, sedclass = remove_outliers(data[:,:3], data[:,3], factor)
data[:,2] = 1 - data[:,2]
tmp_df = pd.DataFrame({'Homogeneity':data[:,2],'sedclass':sedclass})

tmp_df = tmp_df[tmp_df['sedclass'] != 2]


predictors = ['Entropy','Variance','Homogeneity']



classes = ['Sand','Boulders']

standardize = 1

for covtype in ['tied']:
   print 'Working on covariance type %s...' %(covtype,)
   
   for n in [2]:
      print "working on "+predictors[n]
      print "Working on GMM..."
      # split into 50% training, 50% testing
      if standardize==1: # standardize data
         X_train, X_test, y_train, y_test = cross_validation.train_test_split(preprocessing.scale(tmp_df['Homogeneity'].values), tmp_df['sedclass'].values, test_size=0.5, random_state=0)
         tmp_df['Homogeneity'] =preprocessing.scale(tmp_df['Homogeneity'].values)
      else:
         X_train, X_test, y_train, y_test = cross_validation.train_test_split(tmp_df['Homogeneity'].values, tmp_df['sedclass'].values, test_size=0.5, random_state=0)
    
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


      
      bic = g.bic(np.expand_dims(X_train, axis=1))
     
      # test
      y_test_pred = g.predict(np.expand_dims(X_test, axis=1))
      
      y_test_pred[y_test_pred==1] = 3
      y_test_pred[y_test_pred==0] = 1
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
      sand['Homogeneity'].plot.hist(bins=50,color=['r'],ax=ax,alpha=0.6,normed=True,label='Sand')

      boulders['Homogeneity'].plot.hist(bins=50,color=['g'],ax=ax,alpha=0.6,normed=True,label='Boulders')
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
      ax.set_xlabel('Homogeneity')
      ax.legend(fontsize=10)
      plt.setp(plt.xticks()[1], fontsize=7)#, rotation=45)
      plt.setp(plt.yticks()[1], fontsize=7)#, rotation=45)
      plt.title('Non-Bayesian, Accuracy: '+str(test_accuracy)[:4] +'\n Covariance Type: ' + covtype + ', BIC = ' + str(int(bic)))
      
      
###################################################################################################################
#####################Variance + Homogeneity,3 goussicans
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
         X_train, X_test, y_train, y_test = cross_validation.train_test_split(preprocessing.scale(tmp_df[['Variance','Homogeneity']].values), tmp_df['sedclass'].values, test_size=0.5, random_state=0)
         tmp_df['Variance'] = preprocessing.scale(tmp_df[['Variance']].values)
         tmp_df['Homogeneity'] = preprocessing.scale(tmp_df[['Variance']].values)        
      else:
         X_train, X_test, y_train, y_test = cross_validation.train_test_split(tmp_df[['Variance','Homogeneity']].values, tmp_df['sedclass'].values, test_size=0.5, random_state=0)
         tmp_df['Variance'] = preprocessing.scale(tmp_df[['Variance']].values)
         tmp_df['Homogeneity'] = preprocessing.scale(tmp_df[['Variance']].values)  
         
      #initialize the GMM with means
      g = mixture.GaussianMixture(n_components=3, max_iter=100, random_state=0, covariance_type=covtype)
      #g.means_init =  np.array([X_train[y_train == i].mean(axis=0) for i in range(1,len(classes)+1)])

      # fit the model
      g.fit(X_train )
   
      #make sure the means are in order
      order = np.argsort(g.means_[:,0])
      g.means_[:,0] = g.means_[:,0][order]
      g.covariances_[:,0] = g.covariances_[:,0][order]
        
      bic = g.bic(X_train)

      # test
      y_test_pred = g.predict(X_test)+1
      test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
      print "======================================="
      print "test scores: "+predictors[n]
      print test_accuracy

      print(classification_report(y_test_pred.ravel(), y_test.ravel()))

      # show normalized confusion matrix
      cm = confusion_matrix(y_test.ravel(), y_test_pred.ravel())
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print(cm)
      
      
      
      
      color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
    
     # Plot the winner
      splot = plt.subplot(1, 1, 1)
      Y_ = g.predict(X_train)
      
      labels = ['Sand','Gravel','Boulders']
      n = 0
      for i, (mean, cov, color) in enumerate(zip(g.means_, g.covariances_,
                                               color_iter)):
          print cov
          v, w = linalg.eigh(cov)
          if not np.any(Y_ == i):
              continue
          plt.scatter(X_train[Y_ == i, 0], X_train[Y_ == i, 1], .8, color=color,label= labels[n])
    
          # Plot an ellipse to show the Gaussian component
          angle = np.arctan2(w[0][1], w[0][0])
          angle = 180. * angle / np.pi  # convert to degrees
          v = 2. * np.sqrt(2.) * np.sqrt(v)
          ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
          ell.set_clip_box(splot.bbox)
          ell.set_alpha(.5)
          splot.add_artist(ell)
          n+=1
          
      y_train_pred = g.predict(X_train) +1
      train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
      plt.text(0.05, 0.9, 'Train accuracy: %.1f' % train_accuracy,
              transform=splot.transAxes)

      y_test_pred = g.predict(X_test)+1
      test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
      plt.text(0.05, 0.8, 'Test accuracy: %.1f' % test_accuracy,
              transform=splot.transAxes)
      plt.legend()
      plt.xticks(())
      plt.yticks(())
      plt.title('Variance+Homogeneity: full covariance model \n  3 fitted gaussians')
      plt.subplots_adjust(hspace=.35, bottom=.02)
      plt.show()
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
            sigma = np.sqrt(g.covariances_[k][:,0])
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