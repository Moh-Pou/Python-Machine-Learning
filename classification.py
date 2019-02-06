"""
This code evaluates performances of Logistic Regression and Multinomial Naive Bayes on a wine dataset (https://archive.ics.uci.edu/ml/datasets/wine+quality) and predicts wine quality.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import RepeatedStratifiedKFold,cross_val_score

#Logistic Regression.
#Load the dataset from web.
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wines = pd.read_csv(url, sep=";")

#Divide the dataset into X and y.
#X is a matrix of all the features in the dataset excluding wine quality.
X=wines.drop('quality',axis=1) 
#Wine quality can take values between 3-8. 
#However, for this task the code converts these qualities to a binary value and 
#the model predicts whether a specific wine is "high quality". 
#A wine with a quality value of 7 and above is considered high-quality.
y=wines.loc[:,'quality']>=7
lr=LogisticRegression()
#Initialize the Repeated K-fold cross validation object to have 5 folds and 1000 repetitions.
cv_iterator = RepeatedStratifiedKFold(n_splits=5,n_repeats=1000, random_state=42)
scores=cross_val_score(lr, X, y, cv=cv_iterator, scoring="accuracy")
#plot the distribution of scores across the cross-validation runs.
pd.Series(scores).hist(color='black')

#Multinomial Naive Bayes. 
#This is similar to the previous task, except here multinomial NB is used and the wine qualities are not converted to binary.
#Wines with qualities between 5 and 7 are considered high-quality.
from sklearn.naive_bayes import MultinomialNB
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wines = pd.read_csv(url, sep=";")
#Exclude wines whose qualities that are not between 5 and 7.
wines_trimmed=wines.loc[(wines['quality']>=5) & (wines['quality']<=7),:]
X=wines_trimmed.drop('quality',axis=1)
y=wines_trimmed.loc[:,'quality']
mnnb=MultinomialNB()
#Use RepeatedStratifiedKfold cross-validation to evaluate performance of multinomial naive Bayes (NB) on the wine dataset.
cv_iterator = RepeatedStratifiedKFold(n_splits=5,n_repeats=1000, random_state=42)
#Assess the quality of the model using f1_macro. 
#f1_macro is an evaluation based on the f1 score, but it can be used in multiclass (non-binary) problems. 
scores=cross_val_score(mnnb, X, y, cv=cv_iterator, scoring="f1_macro")
pd.Series(scores).hist(color='red')
