"""
Factor Analysis
Factor analysis is useful technique for reducing dimensionality of data. 
It assumes that multiple observed variables have similar patterns of responses because they are all associated with a latent variable.
This code solves the following problem:
Consumers (n = 99) rate how important they consider each of seven qualities when deciding whether or not to buy a 6-pack of beer: 
cost, volume, alcohol percentage, brewery reputation, the color, aroma, and taste.
The rate is between 0 to 100. Perform factor analysis using 2 latent variables to see whether this data can be represented in 2 dimensions.
1- Normalize the beer data using sklearn's 'scale' function.
2- Perform Factor Analysis using 2 components.
3- Observe the factor loadings, which can be accessed via the model's attribute 'components_'.
4- Visualize the factor loadings for each latent variable using a bar chart.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import scale

beer = pd.read_csv("beer.txt", sep="\t", dtype=np.float64)
beer_scaled=pd.DataFrame(scale(beer))
fa=FactorAnalysis(n_components=2,random_state=126).fit(beer_scaled)
#Observe the factor loadings.
print('Observe the factor loadings:')
print(fa.components_)
print(fa.components_.shape)

#Visualize the factor loadings for each latent variable using a bar chart.
print('Visualize the factor loadings for each latent variable using a bar chart')
fig, ax = plt.subplots()
plt.scatter(fa.components_[0,:],fa.components_[1,:])
plt.xlabel('Factor 1')
plt.ylabel('Factor 2')

header=list(beer)
for i, txt in enumerate(header):
    #print(fa.components_[0,i],fa.components_[1,i],txt)
    ax.annotate(txt, (fa.components_[0,i],fa.components_[1,i]))
