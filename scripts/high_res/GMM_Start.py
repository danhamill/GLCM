# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 15:14:34 2016

@author: dan
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report

print(__doc__)

colors = ['navy', 'turquoise', 'darkorange']

def assign_substrate(thing):
    if thing == 0:
        return 'Sand'
    if thing == 1:
        return 'Gravel'
    if thing == 2:
        return 'Boulders'
        
        
def make_ellipses(gmm, ax):
    for n, color in enumerate(colors):
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

data = pd.read_csv(r"C:\workspace\GLCM\new_output\merged_aggregraded_distributions.csv", sep=',').dropna()
sedclass = data[['sedclass']]
data = data[['Homogeneity','Entropy','Variance']]


sedclass = sedclass[np.isfinite(data['Variance'])]
data = data[np.isfinite(data['Variance'])]
# Break up the dataset into non-overlapping training (75%) and testing
# (25%) sets.
skf = StratifiedKFold(n_splits=4)
# Only take the first fold.
train_index, test_index = next(iter(skf.split(data.values, sedclass[sedclass.columns[0]])))
X_train = data.iloc[train_index].dropna().values

y_train = sedclass.iloc[train_index].dropna().values


X_test = data.iloc[test_index].dropna().values
y_test = sedclass.iloc[test_index].dropna().values


n_classes = len(np.unique(y_train))

# Try GMMs using different types of covariances.
estimators = dict((cov_type, GaussianMixture(n_components=n_classes,
                   covariance_type=cov_type, max_iter=100, random_state=0))
                  for cov_type in ['spherical', 'diag', 'tied', 'full'])

n_estimators = len(estimators)

plt.figure(figsize=(3 * n_estimators // 2, 6))
plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05,
                    left=.01, right=.99)


for index, (name, estimator) in enumerate(estimators.items()[2:3]):
    # Since we have class labels for the training data, we can
#     initialize the GMM parameters in a supervised manner.
    estimator.means_init = np.array([X_train[y_train.flatten() == i+1].mean(axis=0) for i in range(n_classes)])
    estimator.verbose = 1
    # Train the other parameters using the EM algorithm.
    estimator.fit(X_train)

    h = plt.subplot(2, n_estimators // 2, index + 1)
    make_ellipses(estimator, h)

    for n, color in enumerate(colors):
        tmp_data = data[sedclass.values == n+1]
        plt.scatter(tmp_data.iloc[:, 0], tmp_data.iloc[:, 1], s=0.8, color=color,
                    label=np.array(['Sand','Gravel','Boulders'],dtype='|S10')[n])
    
        
    # Plot the test data with crosses
    for n, color in enumerate(colors):
        tmp_data = X_test[y_test.flatten() == n+1]
        plt.scatter(tmp_data[:, 0], tmp_data[:, 1], marker='x', color=color)

    y_train_pred = estimator.predict(X_train)
    y_train_pred=  y_train_pred.reshape(1,-1).T
    gravel_accuracy = np.mean(y_train_pred[y_train==2].ravel() == y_train[y_train==2].ravel())
    sand_accuracy = np.mean(y_train_pred[y_train==1].ravel() == y_train[y_train==1].ravel())
    boulders_accuracy = np.mean(y_train_pred[y_train==3].ravel() == y_train[y_train==3].ravel())
    print sand_accuracy, gravel_accuracy, boulders_accuracy
    
    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
    plt.text(0.05, 0.9, 'Train accuracy: %.1f' % train_accuracy,
             transform=h.transAxes)

    y_test_pred = estimator.predict(X_test)
    y_test_pred = y_test_pred.reshape(1, -1).T
    test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
    gravel_accuracy = np.mean(y_test_pred[y_test==2].ravel() == y_test[y_test==2].ravel())
    sand_accuracy = np.mean(y_test_pred[y_test==1].ravel() == y_test[y_test==1].ravel())
    boulders_accuracy = np.mean(y_test_pred[y_test==3].ravel() == y_test[y_test==3].ravel())
    plt.text(0.05, 0.8, 'Test accuracy: %.1f' % test_accuracy,
             transform=h.transAxes)

    plt.xticks(())
    plt.yticks(())
    plt.title(name)

plt.legend(scatterpoints=1, loc='lower right', prop=dict(size=12))


plt.show()

 
cm = confusion_matrix(y_test,y_test_pred +1).astype('float') / confusion_matrix(y_test,y_test_pred +1).sum(axis=1)[:, np.newaxis]
cr = classification_report(y_test,y_test_pred +1)
print cm
print cr
##################################################################################################################################
#from scipy import linalg
#import itertools
#from sklearn import mixture
#color_iter = itertools.cycle(['navy', 'cornflowerblue', 'darkorange'])
#
#def plot_results(X, Y_, means, covariances, index, title):
#    splot = plt.subplot(2, 1, 1 + index)
#    for i, (mean, covar, color) in enumerate(zip(
#            means, covariances, color_iter)):
#        v, w = linalg.eigh(covar)
#        v = 2. * np.sqrt(2.) * np.sqrt(v)
#        u = w[0] / linalg.norm(w[0])
#        # as the DP will not use every component it has access to
#        # unless it needs it, we shouldn't plot the redundant
#        # components.
#        if not np.any(Y_ == i):
#            continue
#        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)
#
#        # Plot an ellipse to show the Gaussian component
#        angle = np.arctan(u[1] / u[0])
#        angle = 180. * angle / np.pi  # convert to degrees
#        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
#        ell.set_clip_box(splot.bbox)
#        ell.set_alpha(0.5)
#        splot.add_artist(ell)
#
#    plt.xlim(-2., 2.)
#    plt.ylim(0., 6.)
#    plt.xticks(())
#    plt.yticks(())
#    plt.title(title)
#
#plot_results(X_train, estimator.predict(X_train), estimator.means_, estimator.covariances_, 0,'Gaussian Mixture')
#dpgmm = mixture.BayesianGaussianMixture(n_components=3,
#                                        covariance_type='full',weight_concentration_prior_type='dirichlet_distribution').fit(X_train)
#plot_results(X_train, dpgmm.predict(X_train), dpgmm.means_, dpgmm.covariances_, 1,
#             'Bayesian Gaussian Mixture with a Dirichlet process prior')
#
#
#
#from matplotlib.colors import LogNorm
#x = np.linspace(-20., 30.)
#y = np.linspace(-20., 40.)
#X, Y = np.meshgrid(x, y)
#XX = np.array([X.ravel(), Y.ravel()]).T
#Z = -estimator.score_samples(XX)
#Z = Z.reshape(X.shape)
#
#CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
#                 levels=np.logspace(0, 3, 10))
#CB = plt.colorbar(CS, shrink=0.8, extend='both')
#plt.scatter(X_train[:, 0], X_train[:, 1], .8)
#
#plt.title('Negative log-likelihood predicted by a GMM')
#plt.axis('tight')
#plt.show()
#
#xs = np.atleast_2d(np.linspace(0, 1, 100)).T
#ys = np.atleast_2d(np.linspace(0, 1, 100)).T
#
#
#
#
#h_data = pd.read_csv(r"C:\workspace\GLCM\new_output\merged_aggregraded_distributions.csv", sep=',').dropna()
#homo = h_data[['Homogeneity','sedclass']]
#ent = h_data[['Entropy','sedclass']]
#var = h_data[['Variance','sedclass']]
#fig ,ax = plt.subplots(nrows=1,ncols=1)
#homo[homo['sedclass'] ==1]['Homogeneity'].plot.hist(ax=ax,bins=50,color='blue',normed=True)
#homo[homo['sedclass'] ==2]['Homogeneity'].plot.hist(ax=ax,bins=50,color='green',normed=True)
#homo[homo['sedclass'] ==3]['Homogeneity'].plot.hist(ax=ax,bins=50,color='red',normed=True)
#
#for i in range(estimator.n_components):
#    pdf = estimator.weights_[i] * stats.norm(estimator.means_[i, 0],
#                                       np.sqrt(estimator.covariances_ [i, 0])).pdf(np.linspace(0,0.8,100))
#    ax.fill(np.linspace(0,0.8,100), pdf, facecolor='gray',
#             edgecolor='none', alpha=0.3)
#    
    