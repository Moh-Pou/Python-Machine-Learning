"""
This code implements a Gaussian Mixture Model (GMM).
It uses the Old Faithful dataset which is a geyser that is found in Yellowstone National Park, Wyoming. 
The code uses a dataset from 1990 that recorded the time between eruptions and the duration of the eruption, both taken in minutes. 
There are 272 observations in total.
"""
from sklearn import cluster, datasets
import numpy as np
import pandas as pd
# URL for the dataset
url = "https://raw.githubusercontent.com/barneygovan/from-data-with-love/master/data/faithful.csv"
data=pd.read_csv(url,skipinitialspace=True)

#Visualize the data.
#plot the distribution of eruption times, waiting times, and then make a scatter plot of both dimensions.
import matplotlib.pyplot as plt
%matplotlib inline
plt.figure(figsize=(15,5))
plt.subplot(131)
plt.hist(data['eruptions'])
plt.ylabel('Eruptions')
plt.subplot(132)
plt.hist(data['waiting'])
plt.ylabel('Waiting Time')
plt.subplot(133)
plt.scatter(data['eruptions'],data['waiting'])
plt.xlabel('Eruptions')
plt.ylabel('Waiting Time')

#Build a GMM.
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=2, random_state=126,).fit(data)
labels = gmm.predict(data)
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.scatter(data['eruptions'],data['waiting'],c=labels)
plt.xlabel('Eruptions')
plt.ylabel('Waiting Time')

#Choosing K.
"""
The Akaike information criterion (AIC) and the Bayesian information criterion (BIC) provide quantitative ways to 
choose the number of clusters that maximize the likelihood of our data while penalizing for increased complexity. 
Scikit-Learn's GMM estimator includes built-in methods that compute both of these. 
Look at the AIC and BIC as a function of the number of GMM components; visualize the AIC and BIC metrics for 10 different models.
"""
n_gaussians = np.arange(1, 10)

# create a Gaussian mixture model for the range 1-10
models = [GaussianMixture(n, random_state=126).fit(data)
          for n in n_gaussians]

# plot the AIC and BIC.
#The optimal number of clusters is the value that minimizes the AIC or BIC. 
plt.subplot(122)
plt.plot(n_gaussians, [m.bic(data) for m in models], label='BIC')
plt.plot(n_gaussians, [m.aic(data) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components');
