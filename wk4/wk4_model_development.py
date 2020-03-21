#############################
# Week 4: Model Development #
#############################
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

###################
# Question #1 c): #
###################
# checking intercept of trained model
lm.intercept_

# checking slope of trained model
lm.coef_

###################
# Question #1 d): #
###################
Yhat = lm.intercept_ + lm.coef_*X

##############################
# Multiple Linear Regression #
##############################

# subsetting
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]

# creating new fit, using multiple predictors
lm.fit(Z, df['price'])

# checking intercept
lm.intercept_

# and coefficients
lm.coef_

###################
# Question #2 a): #
###################
# declaring new regression object
lm2 = LinearRegression()

# training a model using new list of predictors
Z2 = df[['normalized-losses', 'highway-mpg']]

# fitting our model, using vars deemed to be relevant
lm2.fit(Z2, df['price'])

###################
# Question #2 b): #
###################
# checking intercept
lm2.intercept_

# checking size of effects
lm2.coef_


###########################################
# 2) Model Evaluation using Visualization #
###########################################
# importing seabord package
import seaborn as sns

# plotting regression line for price vs. highway-mpg
width = 12
height = 10
#plt.figure(figsize=(width, height))
#sns.regplot(x="highway-mpg", y="price", data=df)
#plt.ylim(0,)

# updating plot to obtain price vs peak-rpm
#plt.figure(figsize=(width, height))
#sns.regplot(x="peak-rpm", y="price", data=df)
#plt.ylim(0,)

################
# Question #3: #
################
# checking correlations
df[['price', 'highway-mpg', 'peak-rpm']].corr()

# stronger negative correlation between price and highway mpg,
# than between price and peak rpm

#################
# Residual Plot #
#################

width = 12
height = 10
# plt.figure(figsize=(width, height))
# sns.residplot(df['highway-mpg'], df['price'])

##############################
# Multiple Linear Regression #
##############################
Yhat = lm2.predict(Z2)

# plt.figure(figsize=(width, height))

# ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
# sns.distplot(Yhat, hist=False, color="b", label="Fitted Values", ax=ax1)

# plt.title('Actual vs Fitted Values for Price')
# plt.xlabel('Price (in dollars)')
# plt.ylabel('Proportion of Cars')

# plt.show()
# plt.close()

###############################################
# Part 3: Polynomial Regression and Pipelines #
###############################################
# defining a function for use in plotting

def PlotPolly(model, independent_variable, dependent_variable, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variable, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Highway-Mpg')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()

x = df['highway-mpg']
y = df['price']

# fitting a cubic function
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
# print(p)

# plotting cubic graph
# PlotPolly(p, x, y, 'highway-mpg')
# np.polyfit(x, y, 3)

################
# Question #4: #
################
# creating an 11 order polynomial model
f = np.polyfit(x, y, 11)
p = np.poly1d(f)
p

# plotting 11th order polynomial graph

# PlotPolly(p, x, y, 'highway-mpg')
# np.polyfit(x, y, 11)

# importing polynomial tool
from sklearn.preprocessing import PolynomialFeatures

# creating polynomialfeatures object, degree 2
pr=PolynomialFeatures(degree=2)
pr

Z_pr = pr.fit_transform(Z)

Z.shape

# transformation adds features, goes from
# 201 samples & 4 features to 201 samples and 15 features
Z_pr.shape


############
# Pipeline #
############
# setting up our pipeline for analysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# create a list of tuples
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model', LinearRegression())]

# inputting the list to our pipeline
pipe=Pipeline(Input)
pipe

# normalizing, transforming, and fitting model using pipeline
# all in one step simultaneously
pipe.fit(Z,y)

# predicting values as usual
ypipe=pipe.predict(Z)

# and printing head of our model
ypipe[0:4]

################
# Question #5: #
################
# standizing and fitting Z
# without transforming it to polynomial

Input=[('scale',StandardScaler()), ('model', LinearRegression())]

# utilizing more restricted pipeline
pipe = Pipeline(Input)
pipe

# fitting model
pipe.fit(Z, y)
ypipe=pipe.predict(Z)

# displaying
ypipe[0:4]

#############################################
# Part 4: Measures for In-Sample Evaluation #
#############################################

#####################################
# Model 1: Simple Linear Regression #
#####################################
# highway_mpg fit
lm.fit(X, Y)

# finding the R^2
#print('The R-square is:', lm.score(X, Y))

# predicting Y values of price using highway mpg
Yhat = lm.predict(X)
#print('The output of the first four predicted values is:', Yhat[0:4])

# using mean squared error function to diagnose mse
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(df['price'], Yhat)
#print('The mean square error of price and predicted value is:', mse)


#######################################
# Model 2: Multiple Linear Regression #
#######################################
# fitting the new model
lm.fit(Z, df['price'])

# Finding the R^2
#print('The R-square is:', lm.score(Z, df['price']))

# predicting the fit for multilinear model
Y_predict_multifit = lm.predict(Z)

# print('The mean square error of price and predicted value using multifit is: ',\
#         mean_squared_error(df['price'], Y_predict_multifit))


###########################
# Model 3: Polynomial Fit #
###########################
from sklearn.metrics import r2_score

r_squared = r2_score(y, p(x))
# print('The R-square value is: ', r_squared)

# print(mean_squared_error(df['price'], p(x)))

##########################################
# Part 5: Prediction and Decision Making #
##########################################
# creating a new input
new_input = np.arange(1, 100, 1).reshape(-1, 1)

# fitting the model
lm.fit(X, Y)
print(lm)

# producing a prediction
yhat=lm.predict(new_input)
yhat[0:5]

# plotting the data
# plt.plot(new_input, yhat)
# plt.show()






























# in order to display plot within window
# plt.show()
