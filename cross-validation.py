"""
This code uses a generated dataset to practice cross-validation and parameter optimization.
First, it runs a linear regression using cross-validation.
Secondly, it runs linear regression using Lasso and cross-validation.
Finally, it runs linear regression with nested cross-validation.
"""

#Run linear regression with 5-fold cross-validation.
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# Generate data
X, y = make_regression(n_samples=100, n_features=100, n_informative=10, random_state=10)

# Iterator setup
cv_iterator = KFold(n_splits=5, shuffle=True, random_state=10)

#Run linear regression
lr = LinearRegression()
lr.fit(X, y)
#use SKLearn's cross_val_score function to assess how well the model generalizes.
cv_score=cross_val_score(lr, X, y, cv=cv_iterator, scoring="neg_mean_squared_error")

#Run linear regression using Lasso and with cross-validation (hyperparameter alpha = 0.5).
from sklearn.linear_model import Lasso
cv_iterator = KFold(n_splits=5, shuffle=True, random_state=10)
llr = Lasso(alpha=0.5)
llr.fit(X, y)
#use SKLearn's cross_val_score function to assess how well the model generalizes.
cv_score=cross_val_score(llr, X, y, cv=cv_iterator, scoring="neg_mean_squared_error")

#Run nested cross-validation while optimizing for parameter alpha.
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error

# Parameter grid
p_grid = {
    "alpha": [0.1, 0.5, 1, 1.5]
}

# CV iterators
inner_cv_iterator = KFold(n_splits=5, shuffle=True, random_state=10)
outer_cv_iterator = KFold(n_splits=5, shuffle=True, random_state=10)

lasso = Lasso()
llr = GridSearchCV(estimator=lasso, param_grid=p_grid, cv=inner_cv_iterator)
#use SKLearn's cross_val_score function to assess how well the model generalizes.
cv_score=cross_val_score(llr, X=X, y=y, cv=outer_cv_iterator,scoring="neg_mean_squared_error")
