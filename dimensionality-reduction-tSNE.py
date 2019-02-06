"""
This code applies tSNE to the digits dataset and transforms it to 2D. 
Digits dataset has about 180 samples per class which justifies perplexity=180 in tSNE. 
Next, it performs K-means clustering into 10 clusters in 2D feature space obtained after tSNE transformation. 
Finally, it calculates sklearn.metrics.adjusted_rand_score between the labels obtained in clustering and the original digit labels.
"""
from sklearn.manifold import TSNE
import sklearn.datasets
import sklearn.cluster
import sklearn.metrics
X,y=sklearn.datasets.load_digits(n_class=10, return_X_y=True)
X_new = TSNE(n_components=2, perplexity=180).fit_transform(X)
kmeans = sklearn.cluster.KMeans(n_clusters=10, random_state=0).fit(X_new)
y_new=kmeans.labels_
score=sklearn.metrics.cluster.adjusted_rand_score(y, kmeans.labels_)

#Visualization.
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.DataFrame({'tSNE 1': X_new[:, 0], 'tSNE 2': X_new[:, 1], 'actual_label': y, 'cluster': y_new})
sns.lmplot(data=df, x='tSNE 1', y='tSNE 2', hue='actual_label', fit_reg=False)
sns.lmplot(data=df, x='tSNE 1', y='tSNE 2', hue='cluster', fit_reg=False)
