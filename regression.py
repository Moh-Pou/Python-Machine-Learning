"""
This code uses the Ames housing dataset. 
The basic goal is to predict housing prices for houses in Ames, Iowa.
First, it loads the training dataset and converts it into a Pandas dataframe.
Next, it plots the data.
Next, it preprocesses the data.
Next, it builds a linear regression model.
Finally, it builds a random forest model.
"""
import pandas as pd
import numpy as np

# Load the ames dataset into a pandas dataframe.
ames_data = pd.read_table("ames_train.csv", sep=",")

#Plot the data
import matplotlib.pyplot as plt
%matplotlib inline
plt.figure(figsize=(10,5))
plt.subplot(221)
ames_data["FirstFlrSF"].hist()
plt.subplot(222)
ames_data["SecondFlrSF"].hist()
plt.subplot(223)
plt.scatter(ames_data["FirstFlrSF"], ames_data["SalePrice"])
plt.subplot(224)
plt.scatter(ames_data["SecondFlrSF"], ames_data["SalePrice"])

#Preprocess the data.
#Create a new column that contains the summation of these two.
ames_data["sqft_sum"]=ames_data["FirstFlrSF"]+ames_data["SecondFlrSF"]

#Build a linear regression model.
#The features are: sqft_sum and Fireplaces (the number of fireplaces in the house).
#Once the model is trained, predict the SalePrice for the same data which was trained on.
from sklearn.linear_model import LinearRegression
X = ames_data.loc[:, ["sqft_sum","Fireplaces"]]
y = ames_data.loc[:, ["SalePrice"]]
lr = LinearRegression()
lr.fit(X, y)
y_preds=lr.predict(X)

#Build a random forest model.
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()
rf.fit(X, y)
y_preds=rf.predict(X)
