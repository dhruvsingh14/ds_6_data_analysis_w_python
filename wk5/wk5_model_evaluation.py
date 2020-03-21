############################
# Week 5: Model Evaluation #
############################
# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#######################
# Reading the Dataset #
#######################
# read url
path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(path)
df.head()

#######################################################
# 1. Linear Regression and Multiple Linear Regression #
#######################################################
# loading linear regression modules
from sklearn.linear_model import LinearRegression

# creating regression object type
lm = LinearRegression()
lm

# single variable prediction: price vs. highway mpg
X = df[['highway-mpg']]
Y = df[['price']]

# fitting model
lm.fit(X, Y)

# outputting prediction
Yhat = lm.predict(X)
Yhat[0:5]

# checking value of intercept from fit
lm.intercept_

# checking value of slope (b)
lm.coef_

###################
# Question #1 a): #
###################
# creating linear regression object
lm = LinearRegression()

###################
# Question #1 b): #
###################
# regressing price on engine size
X = df[['engine-size']]
Y = df[['price']]

# training model
lm.fit(X, Y)

# outputting prediction values
Yhat = lm.predict(X)

# printing first 5 values of Yhat
Yhat[0:5]































# in order to display plot within window
# plt.show()
