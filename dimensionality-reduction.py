"""
Dimensionality reduction and feature selection.
Olivetti faces dataset consists of 10x portaits of 40 individuals. 
Each portrait is a grayscale image with 64x64 pixels, thus the dimensionality of the problem is 4096 if each pixel is considered as a feature.
Apply feature selection to the dataset, identifying top 25% most important features based on the chi-squared criteria (use SelectPercentile).
"""
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import sklearn.datasets
from sklearn.feature_selection import SelectPercentile, chi2
#Read in the dataset.
data = sklearn.datasets.fetch_olivetti_faces()
y=data.target
X=data.data
X_new = SelectPercentile(chi2, percentile=25).fit_transform(X, y)

#Visualize selected features on a 64x64 pixel heatmap for visual aid. 
#Chi2 score of each feature (pixel).
importances = np.array(chi2(X, y)[0])
plt.matshow(importances.reshape(64,64), cmap=plt.cm.hot)

#Only top 25% of the most important features.
min_chi2 = min(chi2(X_new, y)[0])
importances = np.where(importances < min_chi2, np.inf, importances)
plt.matshow(importances.reshape(64,64), cmap=plt.cm.hot)
