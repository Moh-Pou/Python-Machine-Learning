These are Python codes pertaining to supervised and unsupervised machine learning methods. The input files are also in the repository.

1- data-retrieval-preprocessing.py: Loads the dataset into a Pandas dataframe, imputes missing values, creates two new columns, normalizes, and adds a column with a boolean value.

2- regression.py: Loads the dataset into a Pandas dataframe, plots the data, preprocesses the data, builds a linear regression model, and builds a random forest model.

3- cross-validation.py: Runs a linear regression with cross-validation, Lasso regression with cross-validation, and linear regression with nested cross-validation.

4- classification.py: Evaluates performances of Logistic Regression and Multinomial Naive Bayes on a dataset.

5- clustering-K-means.py: Loads the digits dataset into a Pandas data frame, preprocesses the dataset, examines how many clusters are needed using the elbow method and silhouette coefficient, clusters the digits dataset using K-means and evaluates using adjusted rand index.

6- clustering-GMM.py: Loads a dataset, vizualizes it, and builds a Gaussian Mixture Model.

7- factor-analysis.py: Loads a dataset, normalizes it, performs Factor Analysis, and visualizes the factor loadings for each latent variable.

8- feature-selection-chi2.py: Loads Olivetti faces dataset consisting of 10x portaits of 40 individuals, visualizes selected features on a heatmap, applies feature selection to the dataset and identifies top 25% most important features based on the chi-squared criteria.

9- dimensionality-reduction-tSNE.py: Loads the digits dataset, applies tSNE to transform the data to 2 dimensional space, performs K-means clustering on the transformed data, and calculates the scores by comparing the labels obtained in clustering and the original labels.
