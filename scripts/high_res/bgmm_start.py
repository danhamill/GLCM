# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 13:43:32 2016

@author: dan
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import StratifiedKFold
from scipy import stats
import pytablewriter

def cm_markdown(cm,name):
    writer = pytablewriter.MarkdownTableWriter()
    writer.table_name = name + ' confusion matrix'
    writer.header_list = ['index','sand','gracel','boulders']
    index = ['sand','gravel','boulders']
    cm = np.c_[index,cm]
    writer.value_matrix = cm
    writer.write_table()
    
print(__doc__)

colors = ['navy', 'turquoise', 'darkorange']

def assign_substrate(thing):
    if thing == 1:
        return 'Sand'
    if thing == 2:
        return 'Gravel'
    if thing == 3:
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
estimators = dict((cov_type, BayesianGaussianMixture(n_components=n_classes,
                   covariance_type=cov_type, max_iter=100, random_state=0,weight_concentration_prior_type='dirichlet_distribution'))
                  for cov_type in ['spherical', 'diag', 'tied', 'full'])
n_estimators = len(estimators)

plt.figure(figsize=(3 * n_estimators // 2, 6))
plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05,
                    left=.01, right=.99)


for index, (name, estimator) in enumerate(estimators.items()):
    # Since we have class labels for the training data, we can
#     initialize the GMM parameters in a supervised manner.
    #estimator.means_init = np.array([X_train[y_train.flatten() == i+1].mean(axis=0) for i in range(n_classes)])
    #estimator.verbose = 1
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
    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
    plt.text(0.05, 0.9, 'Train accuracy: %.1f' % train_accuracy,
             transform=h.transAxes)

    y_test_pred = estimator.predict(X_test)
    test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100


    plt.text(0.05, 0.8, 'Test accuracy: %.1f' % test_accuracy,
             transform=h.transAxes)

    plt.xticks(())
    plt.yticks(())
    plt.title(name)
    cm = confusion_matrix(y_test,y_test_pred +1).astype('float') / confusion_matrix(y_test,y_test_pred +1).sum(axis=1)[:, np.newaxis]
    cm_markdown(cm,name)
plt.legend(scatterpoints=1, loc='lower right', prop=dict(size=12))


plt.show()
        
 
        



