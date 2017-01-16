
from __future__ import division
import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from sklearn import mixture
from sklearn import cross_validation
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing

# =========================================================
def remove_outliers(X, y, k):
   """
   simple outlier removal based on deviation from mean
   """
   mu, sigma = np.mean(X, axis=0), np.std(X, axis=0, ddof=1)
   index = np.all(np.abs((X - mu) / sigma) < k, axis=1)
   return X[index], y[index]

#===========================================================================
data = np.genfromtxt(r"C:\workspace\GLCM\new_output\merged_aggregraded_distributions.csv", delimiter=',', skip_header=1)

data = data[~np.isnan(data).any(axis=1)]

# for removing outliers
factor=3 #smaller the number, the more ruthless the pruning

data, sedclass = remove_outliers(data[:,:3], data[:,3], factor)
data[:,2] = 1 - data[:,2]
predictors = ['Entropy','Variance','Homogeneity']

#sedclass = data[:,3]

classes = ['sand', 'gravel', 'boulder']

standardize = 1

for covtype in ['full', 'tied', 'diag', 'spherical']:
   print 'Working on covariance type %s...' %(covtype,)
   plt.figure(figsize = (12,10))
   for n in [0,1,2]:
      print "working on "+predictors[n]
      print "Working on BGMM..."
      # split into 50% training, 50% testing
      if standardize==1: # standardize data
         X_train, X_test, y_train, y_test = cross_validation.train_test_split(preprocessing.scale(data[:,n]), sedclass, test_size=0.5, random_state=0)
      else:
         X_train, X_test, y_train, y_test = cross_validation.train_test_split(data[:,n], sedclass, test_size=0.5, random_state=0)

      bg = mixture.BayesianGaussianMixture(n_components=3, max_iter=100, random_state=0, covariance_type=covtype)
      bg.means_init =  np.array([X_train[y_train == i].mean(axis=0) for i in range(1,len(classes)+1)])
      bg.means_init =np.expand_dims(bg.means_init, axis=1) 

      bg.fit(np.expand_dims(X_train, axis=1) )

      #make sure the means are in order
      order = np.argsort(np.squeeze(bg.means_))
      bg.means_ = bg.means_[order]
      try:
         bg.covariances_ = bg.covariances_[order]
      except:
         pass
      bg.weights_ = bg.weights_[order]

      # test
      y_test_pred = bg.predict(np.expand_dims(X_test, axis=1))+1
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

      plt.subplot(2,3,n+1)
      plt.bar(bins[1:], d/np.max(d), color='k', alpha=0.5, width=0.05)

      col='rgb'
      for k in range(len(classes)):
         mu = bg.means_[k]
         try:
            sigma = np.sqrt(bg.covariances_[k])
         except:
            sigma = np.sqrt(bg.covariances_)
         yy  = np.squeeze(bg.weights_[k]*stats.norm.pdf(x,mu,sigma))
         plt.plot(x,yy/np.max(yy), c=col[k], linewidth=1, label=classes[k])
      plt.xlabel(predictors[n])
      plt.legend(fontsize=6)
      plt.setp(plt.xticks()[1], fontsize=7)#, rotation=45)
      plt.setp(plt.yticks()[1], fontsize=7)#, rotation=45)
      plt.title('Bayesian, Accuracy: '+str(test_accuracy)[:4])
      print "Working on GMM..."
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

      plt.subplot(2,3,n+4)
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
      plt.xlabel(predictors[n])
      plt.legend(fontsize=6)
      plt.setp(plt.xticks()[1], fontsize=7)#, rotation=45)
      plt.setp(plt.yticks()[1], fontsize=7)#, rotation=45)
      plt.title('non-Bayesian, Accuracy: '+str(test_accuracy)[:4])
      plt.suptitle('Covariance type: ' + covtype)



#################################
#============ entropy and variance
print 'Entropy and variance GMM..'
# split into 50% training, 50% testing
X_train, X_test, y_train, y_test = cross_validation.train_test_split(data[:,(0,1)], sedclass, test_size=0.5, random_state=0)

#initialize the GMM with means
g = mixture.GaussianMixture(n_components=3, max_iter=100, random_state=0, covariance_type='full')
g.means_init =  np.array([X_train[y_train == i].mean(axis=0) for i in range(1,len(classes)+1)])

# fit the model
g.fit(X_train)
   
#make sure the means are in order
order = np.argsort(g.means_[:,0])
g.means_[:,0] = g.means_[:,0][order]
g.covariances_[:,0] = g.covariances_[:,0][order]

order = np.argsort(g.means_[:,1])
g.means_[:,1] = g.means_[:,1][order]
g.covariances_[:,1] = g.covariances_[:,1][order]

# test
y_test_pred = g.predict(X_test)+1
test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
print test_accuracy

print(classification_report(y_test_pred.ravel(), y_test.ravel()))


# http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_pdf.html
xmin = np.min(X_train, axis=0)
xmax = np.max(X_train, axis=0)
x = np.linspace(xmin[0],xmax[0],1000)
y = np.linspace(xmin[1],xmax[1],1000)

X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -g.score_samples(XX)
Z = Z.reshape(X.shape)

plt.figure(figsize=(12,6))
plt.subplot(121)
CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(X_train[:, 0], X_train[:, 1], .8)

plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.xlabel(predictors[0])
plt.ylabel(predictors[1])
#plt.show()

print 'Entropy and variance BGMM..'
#initialize the GMM with means
bg = mixture.BayesianGaussianMixture(n_components=3, max_iter=100, random_state=0, covariance_type='full')
bg.means_init =  np.array([X_train[y_train == i].mean(axis=0) for i in range(1,len(classes)+1)])

# fit the model
bg.fit(X_train)
   
#make sure the means are in order
order = np.argsort(bg.means_[:,0])
bg.means_[:,0] = bg.means_[:,0][order]
bg.covariances_[:,0] = bg.covariances_[:,0][order]

order = np.argsort(bg.means_[:,1])
bg.means_[:,1] = bg.means_[:,1][order]
bg.covariances_[:,1] = bg.covariances_[:,1][order]

# test
y_test_pred = bg.predict(X_test)+1
test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
print test_accuracy

print(classification_report(y_test_pred.ravel(), y_test.ravel()))

Z = -bg.score_samples(XX)
Z = Z.reshape(X.shape)

plt.subplot(122)
CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(X_train[:, 0], X_train[:, 1], .8)

plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.xlabel(predictors[0])
plt.ylabel(predictors[1])
plt.show()



print 'Entropy and homogeneity GMM..'
#################################
#============ entropy and homogeneity
# split into 50% training, 50% testing
X_train, X_test, y_train, y_test = cross_validation.train_test_split(data[:,(0,2)], sedclass, test_size=0.5, random_state=0)

#initialize the GMM with means
g = mixture.GaussianMixture(n_components=3, max_iter=100, random_state=0, covariance_type='full')
g.means_init =  np.array([X_train[y_train == i].mean(axis=0) for i in range(1,len(classes)+1)])

# fit the model
g.fit(X_train)
   
#make sure the means are in order
order = np.argsort(g.means_[:,0])
g.means_[:,0] = g.means_[:,0][order]
g.covariances_[:,0] = g.covariances_[:,0][order]

order = np.argsort(g.means_[:,1])
g.means_[:,1] = g.means_[:,1][order]
g.covariances_[:,1] = g.covariances_[:,1][order]

# test
y_test_pred = g.predict(X_test)+1
test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
print test_accuracy

print(classification_report(y_test_pred.ravel(), y_test.ravel()))


# http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_pdf.html
xmin = np.min(X_train, axis=0)
xmax = np.max(X_train, axis=0)
x = np.linspace(xmin[0],xmax[0],1000)
y = np.linspace(xmin[1],xmax[1],1000)

X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -g.score_samples(XX)
Z = Z.reshape(X.shape)

plt.figure()
plt.subplot(121)
CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(X_train[:, 0], X_train[:, 1], .8)

plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.xlabel(predictors[0])
plt.ylabel(predictors[2])
#plt.show()

print 'Entropy and homogeneity BGMM..'
#initialize the GMM with means
bg = mixture.BayesianGaussianMixture(n_components=3, max_iter=100, random_state=0, covariance_type='full')
bg.means_init =  np.array([X_train[y_train == i].mean(axis=0) for i in range(1,len(classes)+1)])

# fit the model
bg.fit(X_train)
   
#make sure the means are in order
order = np.argsort(bg.means_[:,0])
bg.means_[:,0] = bg.means_[:,0][order]
bg.covariances_[:,0] = bg.covariances_[:,0][order]

order = np.argsort(bg.means_[:,1])
bg.means_[:,1] = bg.means_[:,1][order]
bg.covariances_[:,1] = bg.covariances_[:,1][order]

# test
y_test_pred = bg.predict(X_test)+1
test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
print test_accuracy

print(classification_report(y_test_pred.ravel(), y_test.ravel()))

Z = -bg.score_samples(XX)
Z = Z.reshape(X.shape)

plt.subplot(122)
CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(X_train[:, 0], X_train[:, 1], .8)

plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.xlabel(predictors[0])
plt.ylabel(predictors[2])
plt.show()




print 'Homogeneity and variance GMM..'
#################################
#============ variance and homogeneity
# split into 50% training, 50% testing
X_train, X_test, y_train, y_test = cross_validation.train_test_split(data[:,(1,2)], sedclass, test_size=0.5, random_state=0)

#initialize the GMM with means
g = mixture.GaussianMixture(n_components=3, max_iter=100, random_state=0, covariance_type='full')
g.means_init =  np.array([X_train[y_train == i].mean(axis=0) for i in range(1,len(classes)+1)])

# fit the model
g.fit(X_train)
   
#make sure the means are in order
order = np.argsort(g.means_[:,0])
g.means_[:,0] = g.means_[:,0][order]
g.covariances_[:,0] = g.covariances_[:,0][order]

order = np.argsort(g.means_[:,1])
g.means_[:,1] = g.means_[:,1][order]
g.covariances_[:,1] = g.covariances_[:,1][order]

# test
y_test_pred = g.predict(X_test)+1
test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100

print "======================================="
print test_accuracy

print(classification_report(y_test_pred.ravel(), y_test.ravel()))


# http://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_pdf.html
xmin = np.min(X_train, axis=0)
xmax = np.max(X_train, axis=0)
x = np.linspace(xmin[0],xmax[0],1000)
y = np.linspace(xmin[1],xmax[1],1000)

X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -g.score_samples(XX)
Z = Z.reshape(X.shape)

plt.figure()
plt.subplot(121)
CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(X_train[:, 0], X_train[:, 1], .8)

plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.xlabel(predictors[1])
plt.ylabel(predictors[2])
#plt.show()

print 'Homogeneity and variance BGMM..'
#initialize the GMM with means
bg = mixture.BayesianGaussianMixture(n_components=3, max_iter=100, random_state=0, covariance_type='full')
bg.means_init =  np.array([X_train[y_train == i].mean(axis=0) for i in range(1,len(classes)+1)])

# fit the model
bg.fit(X_train)
   
#make sure the means are in order
order = np.argsort(bg.means_[:,0])
bg.means_[:,0] = bg.means_[:,0][order]
bg.covariances_[:,0] = bg.covariances_[:,0][order]

order = np.argsort(bg.means_[:,1])
bg.means_[:,1] = bg.means_[:,1][order]
bg.covariances_[:,1] = bg.covariances_[:,1][order]

# test
y_test_pred = bg.predict(X_test)+1
test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
print test_accuracy

print(classification_report(y_test_pred.ravel(), y_test.ravel()))

Z = -bg.score_samples(XX)
Z = Z.reshape(X.shape)

plt.subplot(122)
CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(X_train[:, 0], X_train[:, 1], .8)

plt.title('Negative log-likelihood predicted by a BGMM')
plt.axis('tight')
plt.xlabel(predictors[1])
plt.ylabel(predictors[2])
plt.show()



print 'Entropy, Homogeneity and variance GMM..'
#################################
#============ entropy, variance and homogeneity
# split into 50% training, 50% testing
X_train, X_test, y_train, y_test = cross_validation.train_test_split(data[:,(0,1,2)], sedclass, test_size=0.5, random_state=0)

#initialize the GMM with means
g = mixture.GaussianMixture(n_components=3, max_iter=100, random_state=0, covariance_type='full')
g.means_init =  np.array([X_train[y_train == i].mean(axis=0) for i in range(1,len(classes)+1)])

# fit the model
g.fit(X_train)
   
#make sure the means are in order
order = np.argsort(g.means_[:,0])
g.means_[:,0] = g.means_[:,0][order]
g.covariances_[:,0] = g.covariances_[:,0][order]

order = np.argsort(g.means_[:,1])
g.means_[:,1] = g.means_[:,1][order]
g.covariances_[:,1] = g.covariances_[:,1][order]

# test
y_test_pred = g.predict(X_test)+1
test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
print test_accuracy

print(classification_report(y_test_pred.ravel(), y_test.ravel()))




