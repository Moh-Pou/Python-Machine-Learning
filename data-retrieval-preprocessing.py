"""
The Iris dataset includes 150 observations of flowers, with petal lengths and widths, and sepal lengths and widths. 
The basic goal of this code is to preprocess the iris dataset in preparation for a machine learning algorithm.
First, it loads the dataset into a Pandas dataframe. 
Next, it imputes missing values in one of the columns. 
Next, it assigns the average value in the column to the missing data points.
Next, it creates two new columns that will approximate the sepal and petal sizes: One will equal the petal length multiplied by the petal width, and the other will equal the sepal length multiplied by the sepal width.
Next, it normalizes the sepal and petal sizes.
Finally, it adds a column with a boolean value representing whether a flower belongs to the setosa species.
"""

import pandas as pd
import numpy as np

#Load the dataset into a Pandas dataframe.
iris_data=pd.read_table("Iris.csv", sep=",") 

#Impute missing data
def impute_with_mean(df, column_name):
    """
    Accepts a Pandas data frame and a column name as input.    
    Returns a new Pandas data frame with missing data points in the column
    replaced with the mean value for that column.
    """
    df[column_name]=df[column_name].fillna(df[column_name].mean())
    return df

iris_data = impute_with_mean(iris_data, "PetalLengthCm") 

#Create two new columns that approximate the sepal and petal sizes.
iris_data["sepal_size"]=iris_data["SepalLengthCm"]*iris_data["SepalWidthCm"]
iris_data["petal_size"]=iris_data["PetalLengthCm"].multiply(iris_data["PetalWidthCm"])

#Normalize sizes so that different features are on the same scales. 
#Normalize by subtracting the mean from each column and dividing by the standard deviation.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
iris_data["sepal_size_normalized"]=scaler.fit_transform(iris_data[['sepal_size']])
iris_data["petal_size_normalized"]=scaler.fit_transform(iris_data[['petal_size']])

#Add a boolean column
#Is a given flower from the setosa species?
iris_data["is_setosa"]=iris_data["Species"]=="Iris-setosa"
