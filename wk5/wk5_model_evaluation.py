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
path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/module_5_auto.csv'
df = pd.read_csv(path)

# outputting csv dataset
df.to_csv('module_5_auto.csv')

# sticking to numeric data
df=df._get_numeric_data()
df.head()

# installing ipython packages
from IPython.display import display
# from IPython.html import widgets
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual

##########################
# Functions for plotting #
##########################

def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)
    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')

    plt.show()
    plt.close()

def PollyPlot(xtrain, xtest, y_train, y_test, lr, poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    # training data testing
    # testing data
    # lr: linear regression object
    # poly_transform: polynomial transformation object

    xmax=max([xtrain.values.max(), xtest.values.max()])

    xmin=min([xtrain.values.min(), xtest.values.min()])

    x=np.arange(xmin, xmax, 0.1)

    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()

################################
# Part 1: Training and Testing #
################################
# creating data frame for target var
y_data = df['price']

# and dropping it from dataframe
x_data=df.drop('price', axis=1)

# getting ready to train our model

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=1)

# print("number of test samples :", x_test.shape[0])
# print("number of training samples:", x_train.shape[0])

#################
# Question #1): #
#################
# creating testing and training datasets
x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(x_data, y_data, test_size=0.40, random_state=0)

# getting ready to regress, and train model
from sklearn.linear_model import LinearRegression

lre=LinearRegression()

# using horsepower to predict our outcome
# and to determine our fit
lre.fit(x_train[['horsepower']], y_train)

# calculating R^2 on our test data aka. prediction
# print(lre.score(x_test[['horsepower']], y_test))

# comparing R^2, test vs train, R^2 is smaller
# print(lre.score(x_train[['horsepower']], y_train))

#################
# Question #2): #
#################
# finding R^2 on the test data

# creating new test and training datasets - training size = 90%
x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(x_data, y_data, test_size=0.10, random_state=1)

# regression object for calculating R^2
lre_2 = LinearRegression()

# fit is always created using training on training data
# can be scored using test or training data
lre_2.fit(x_train_2[['horsepower']], y_train_2)

# checking test score for R^2
lre_2.score(x_test_2[['horsepower']], y_test_2)

##########################
# Cross-validation Score #
##########################
# employing cross validation techniques
from sklearn.model_selection import cross_val_score

# evaluating score with 4 folds
Rcross = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4)

# default scoring uses R^2
Rcross

# checking descriptives for R^2, eg mean, s.d
# print("The mean of the folds are", Rcross.mean(), "and the standard deviation is", Rcross.std())

# can also using -mse for scoring
-1 * cross_val_score(lre, x_data[['horsepower']], y_data, cv=4, scoring='neg_mean_squared_error')

#################
# Question #3): #
#################
# calculating average R^2 using 2 folds
Rcross_2 = cross_val_score(lre, x_data[['horsepower']], y_data, cv=2)

# checking R^2 for each fold
Rcross_2

# printing avg R^2 of second fold, using horsepower as a feature
# print("The mean of the folds are", Rcross_2.mean(), "and the standard deviation is", Rcross_2.std())

# moving to prediction
from sklearn.model_selection import cross_val_predict

# using predictor, target and oject to predict using cross validation technique
yhat = cross_val_predict(lre,x_data[['horsepower']], y_data,cv=4)
yhat[0:5]

#########################################################
# Part 2: Overfitting, Underfitting and Model Selection #
#########################################################
# creating reg object
# fitting using multiple - relevant predictors (prev lesson - descriptives)
lr = LinearRegression()
lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)

# training model with multiple predictors
# using training data
yhat_train = lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
yhat_train[0:5]

# now using test data to predict model
yhat_test = lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
yhat_test[0:5]

# getting ready to check model performance by comparing
import seaborn as sns

# checking training model performance
Title = 'Distribution Plot of Predicted Value Using Training Data vs Training Data Distribution'
# DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)

# checking test model performance
Title = 'Distribution Plot of Predicted Value Using Test Data vs Test Data Distribution'
# DistributionPlot(y_test, yhat_test, "Actual Values (Test)", "Predicted Values (Test)", Title)

# evaluating fit
from sklearn.preprocessing import PolynomialFeatures

###############
# Overfitting #
###############
# now using 55 % of the data for testing
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45, random_state=0)

# now constructing a degree 5 polynomial, for polynomial regression
# using single variable
pr = PolynomialFeatures(degree=5)
x_train_pr = pr.fit_transform(x_train[['horsepower']])
x_test_pr = pr.fit_transform(x_test[['horsepower']])
pr

# training polynomial regression model, in one variable
poly = LinearRegression()
poly.fit(x_train_pr, y_train)

# printing first 5 values of predicted model - prices
yhat = poly.predict(x_test_pr)
yhat[0:5]

# comparing first five predicted values to actual target vals
# print("Predicted values:", yhat[0:4])
# print("True values:", y_test[0:4].values)

# plotting training and testing data side by side
# PollyPlot(x_train[['horsepower']], x_test[['horsepower']], y_train, y_test, poly, pr)
# plt.show()

# evaluating training model performance by scoring
poly.score(x_train_pr, y_train)

# evaluating test model performance by scoring
poly.score(x_test_pr, y_test)
# negative R^2 is indicative of overfitting

# getting ready to loop through R-squares and store in array

Rsqu_test = []

order = [1, 2, 3, 4]
for n in order:
    pr = PolynomialFeatures(degree=n)

    x_train_pr = pr.fit_transform(x_train[['horsepower']])

    x_test_pr = pr.fit_transform(x_test[['horsepower']])

    lr.fit(x_train_pr, y_train)

    Rsqu_test.append(lr.score(x_test_pr, y_test))

plt.plot(order, Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')
plt.text(3, 0.75, 'Maximum R^2')
# plt.show()

# I LOVE this function for the way it exemplifies fit
# polynomial function: order vs degree, keep in mind

def f(order, test_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
    pr = PolynomialFeatures(degree=order)
    x_train_pr = pr.fit.transform(x_train[['horsepower']])
    x_test_pr = pr.fit.transform(x_train[['horsepower']])
    poly = LinearRegression()
    poly.fit(x_train_pr, y_train)
    PollyPlot(x_train[['horsepower']])

# interact(f, order=(0, 6, 1), test_data=(0.05, 0.95, 0.05))

# seems we'll have to forego the widget feature for now
# in favor of python3 and atom

##################
# Question #4a): #
##################
# creating polynomial features object
pr1 =PolynomialFeatures(degree=2)

##################
# Question #4b): #
##################
# using polynomial transformation, on test and training data sets
# using x_train and x_test from the last split
x_train_pr1 = pr1.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
x_test_pr1 = pr1.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])

##################
# Question #4c): #
##################
# checking dimensions
x_train_pr1.shape
x_test_pr1.shape

# new feature has 110 samples in training dataset, w/ 15 features post-transformation, some squared
# new feature has 91 samples in test dataset, w/ 15 features post-transformation, some squared

##################
# Question #4d): #
##################
# creating polynomial regression object
poly1 = LinearRegression()

# training new model using training data target, and polynomial features
poly1.fit(x_train_pr1, y_train)

##################
# Question #4e): #
##################
# predicting outcome values
yhat_1 = poly1.predict(x_test_pr1)

# checking test model performance
Title1 = 'Distribution Plot of Predicted Value Using Test Data vs Distribution'
# DistributionPlot(y_test, yhat_1, "Actual Values (Test)", "Predicted Values (Test)", Title1)

##################
# Question #4f): #
##################
# eyeballing

# predicted vals less accurate at around 10 k price mark
# and at around 30 - 40 k price mark

############################
# Part 3: Ridge regression #
############################
# reformulating our polynomial features variables
pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg', 'normalized-losses', 'symboling']])
x_test_pr=pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg', 'normalized-losses', 'symboling']])

# importing ridge regression models from sci kit learn
from sklearn.linear_model import Ridge

# a new novel construction in the data science world
RidgeModel=Ridge(alpha=0.1)

# fitting this model
RidgeModel.fit(x_train_pr, y_train)

# also, predicting target values
yhat = RidgeModel.predict(x_test_pr)

# comparing first few predicted vals to test set by hand
# print('predicted:', yhat[0:4])
# print('test set :', y_test[0:4].values)

# running the same r-squared test
Rsqu_test = []
Rsqu_train = []
dummy1 = []
ALFA = 10 * np.array(range(0,1000))
for alfa in ALFA:
    RidgeModel = Ridge(alpha=alfa)
    RidgeModel.fit(x_train_pr, y_train)
    Rsqu_test.append(RidgeModel.score(x_test_pr, y_test))
    Rsqu_train.append(RidgeModel.score(x_train_pr, y_train))

# setting up plotting features
width = 12
height = 10
plt.figure(figsize=(width, height))

plt.plot(ALFA, Rsqu_test, label='validation data ')
plt.plot(ALFA, Rsqu_train, 'r', label='training Data ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()
# plt.show()

#################
# Question #5): #
#################
# performing ridge regression
# calculating the R^2 score necessary

# reformulating our polynomial features variables
pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg', 'normalized-losses', 'symboling']])
x_test_pr=pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg', 'normalized-losses', 'symboling']])

# a new novel construction in the data science world
RidgeModel=Ridge(alpha=10)

# fitting this model
RidgeModel.fit(x_train_pr, y_train)

# also, predicting target values
yhat = RidgeModel.predict(x_test_pr)

# running the same r-squared test
Rsqu_test = []
Rsqu_train = []
dummy1 = []
ALFA = 10 * np.array(range(0,1000))
for alfa in ALFA:
    RidgeModel = Ridge(alpha=alfa)
    RidgeModel.fit(x_train_pr, y_train)
    Rsqu_test.append(RidgeModel.score(x_test_pr, y_test))
    Rsqu_train.append(RidgeModel.score(x_train_pr, y_train))

# setting up plotting features
width = 12
height = 10
plt.figure(figsize=(width, height))

plt.plot(ALFA, Rsqu_test, label='validation data ')
plt.plot(ALFA, Rsqu_train, 'r', label='training Data ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()
# plt.show()

#######################
# Part 4: Grid Search #
#######################
# importing grid search
from sklearn.model_selection import GridSearchCV

# creating a dictionary of parameter values
parameters1= [{'alpha': [0.001,0.1,1, 10, 100, 1000, 10000, 100000, 1000000]}]

# creating a new ridge regions object
RR=Ridge()
RR

# creating the ridge grid search object
Grid1 = GridSearchCV(RR, parameters1, cv=4)

# fitting the model
Grid1.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)

# grid search locates the best parameter values and optimizes accordingly
BestRR=Grid1.best_estimator_
BestRR

# testing our model, on test data
BestRR. score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_test)

#################
# Question #6): #
#################

# creating a dictionary of parameter values
parameters2= [{'alpha': [0.001,0.1,1, 10, 100, 1000, 10000, 100000, 1000000],
                'normalize': [True, False]}]

# creating the ridge grid search object
Grid2 = GridSearchCV(Ridge(), parameters2, cv=4)

# fitting the model using additional parameter for normalization
Grid2.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)

# grid search locates the best parameter values and optimizes accordingly
Grid2.best_estimator_

#########
# Quiz: #
#########

# q2
x = cross_val_score(lre, x_data, y_data, cv=2)







































# in order to display plot within window
# plt.show()
