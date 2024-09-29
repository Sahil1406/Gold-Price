# IMPORTING ALL THE DEPENDENCIES 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

# IMPORTING THE DATASET 

dataset = pd.read_csv('gold_price_data.csv')

# print(dataset.head()) # --> First 5 rows of the dataset

# print(dataset.tail()) --> Last 5 rows of the dataset

# print(dataset.shape) --> Number of rows and columns\

# print(dataset.info()) --> More info about the dataset

# print(dataset.isnull().sum()) --> Checking for null values

# print(dataset.describe()) --> Statistical values of the Data 

# CHECKING THE CORRELATION THROUGH PLOTS

dataset_nodate = dataset.drop('Date', axis=1)
correlation = dataset_nodate.corr()
plt.figure(figsize=(8,8))
# sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')


# print(correlation['GLD'])

# CHECKING THE DISTRIBUTION OF GOLD PRICE

# sns.displot(dataset['GLD'], color='green')

# SPLITTING THE DATASET FROM THE LABELS

X = dataset.drop(['Date', 'GLD'], axis=1)

Y = dataset['GLD']

# print(X)
# print(Y)

# SPLITTING THE DATA INTO TRAINING AND TESTING PARTS


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size=0.8, random_state=1)

# print(X.shape, X_test.shape, X_train.shape)

# TRAINING THE MODEL

regressor = RandomForestRegressor(n_estimators = 100)
regressor.fit(X_train, Y_train)

# MODEL EVALUATION

# Evaluating for Training data
training_data_prediction = regressor.predict(X_train)
error_score = metrics.r2_score(training_data_prediction, Y_train)
print('R Squared Mean for training data is:', error_score)

# Evaluating for Testing data
testing_data_prediction = regressor.predict(X_test)
error_score = metrics.r2_score(testing_data_prediction, Y_test)
print('R Squared Mean for testing data is:', error_score)