"""
This codes uses the digits dataset which represents images of handwritten digits. 
Each image is a 32x32 bitmap. To create a numerical dataset representing the images, 
each image is divided into non-overlapping blocks of 4x4, and the number of set pixels in each block is counted.
Thus, each image is represented by an 8x8 matrix of integers in the range 0-16. Thus, each sample in the dataset has 64 attributes. 
There are a total of 1,797 samples in the dataset. Therefore, the digits dataset should be loaded into a Pandas dataframe with 1,797 rows and 64 columns.
The code works as follows:
First, it loads the digits dataset into a Pandas data frame.
Next, it preprocesses the digits dataset.
Next, it examines how many clusters are needed using the elbow method and silhouette coefficient.
Finally, it clusters the digits dataset using K-means and evaluate using adjusted rand index.
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline

#Load the digits dataset into a Pandas dataframe.
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes"
digits = pd.read_csv(url,header=None)
labels = digits.iloc[:,64]
digits = digits.drop(digits.columns[64],axis=1)

#Preprocess the digits dataset.
#Standardize the data so that each column has a mean of (or very close to) zero and 
#a standard deviation of (or very close to) 1.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(digits)
digits=scaler.transform(digits)
digits=pd.DataFrame(digits)

#Examine how many clusters are needed using the elbow method and silhouette coefficient.
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import silhouette_score
plt.figure(figsize=(10,5))
"""
The elbow method consists of visualizing the sum of squared distances between samples and their closest cluster center. 
This is done by creating a scatter plot where the number of clusters used in K-means is provided on the X axis and the sum of squared distances is provided on the Y axis. 
If there is a clear inflection point, that can be an indicator of the ideal number of clusters.
"""
#Create an elbow plot to examine whether there is a clear inflection point.
def plot_elbow(dataset, Min_Cluster, max_clusters):
    """Plot elbow curve for k-means."""
    inertias = []
    for i in range(Min_Cluster, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=126)
        kmeans.fit(dataset)
        inertias.append(kmeans.inertia_)
    plt.subplot(121)
    plt.plot(range(Min_Cluster, max_clusters + 1), inertias)
    plt.title("Elbow Plot")
    plt.xlabel("K")
    plt.ylabel("SSD")
    return inertias
#Run the K-means algorithm on the digits dataset 24 times and create an elbow plot to look for a clear inflection point.
ssds=plot_elbow(digits,2,25)

#Use silhouette_score function to calculate Silhouette Coefficient (SC).
def calc_silhouette(dataset, n):
    kmeans = KMeans(n_clusters=n, random_state=126).fit(dataset)
    score = silhouette_score(dataset, kmeans.labels_)
    return score
#Explore dependence of SC on the number of clusters in K-means algorithm. 
scores = {n: calc_silhouette(digits, n) for n in range(2, 26)}
plt.subplot(122)
plt.plot(
    list(scores.keys()),
    list(scores.values())
)

#Cluster the digits dataset using K-means and evaluate using adjusted rand index (ARI).
#Set the number of clusters to 10 (there are 10 digits).
KMeans(n_clusters=10, random_state=126)
#Evaluate clusters using the ARI.
from sklearn.metrics import adjusted_rand_score
model=KMeans(n_clusters=10, random_state=126)
model.fit(digits)
labels_pred=model.predict(digits)
score=adjusted_rand_score(labels, labels_pred)
print(score)
