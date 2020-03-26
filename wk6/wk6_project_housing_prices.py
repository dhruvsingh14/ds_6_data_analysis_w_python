####################################
# Week 6: Project - Housing Prices #
####################################
# importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

##########################
# 1.0 Importing the Data #
##########################
# load csv
file_name = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/coursera/project/kc_house_data_NaN.csv'
df = pd.read_csv(file_name)
df.head()

##############
# Question 1 #
##############
# displaying the data types for each column
df.dtypes

# using describe to get summary statistics
df.describe()

######################
# 2.0 Data Wrangling #
######################

##############
# Question 2 #
##############
# dropping unnecessary columns
df.drop(['id', 'Unnamed: 0'], axis=1, inplace=True)

# using describe to get summary statistics
df.describe()

# checking missing values for bathroom and bedroom columns
# print("number of Nan values for the column bedrooms :", df['bedrooms'].isnull().sum())
# print("number of Nan values for the column bedrooms :", df['bathrooms'].isnull().sum())

# checking average number of bedrooms
mean=df['bedrooms'].mean()
# imputing in mean for nan values of bedrooms column
df['bedrooms'].replace(np.nan, mean, inplace=True)

# checking average number of bathrooms
mean=df['bathrooms'].mean()
# imputing in mean for nan values of bathrooms column
df['bathrooms'].replace(np.nan, mean, inplace=True)

# confirming missing values have been replaced
#print("number of Nan values for the column bedrooms :", df['bedrooms'].isnull().sum())
#print("number of Nan values for the column bedrooms :", df['bathrooms'].isnull().sum())

#################################
# 3.0 Exploratory data analysis #
#################################

##############
# Question 3 #
##############
# checking unique value counts for floor variable
df['floors'].value_counts()

# converting to a dataframe
df2 = df['floors'].value_counts().to_frame()
df2

##############
# Question 4 #
##############
# plotting boxplot
sns.boxplot(x = "waterfront", y= "price", data = df)
# plt.show()

##############
# Question 5 #
##############
# checking slr: single var regression
sns.regplot(x = "sqft_above", y= "price", data = df)
# plt.show()
# sqft_above is positively correlated with price

# checking correlation values
df.corr()['price'].sort_values()

###############################
# Module 4: Model Development #
###############################

# importing libraries
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# fitting a model
X = df[['long']]
Y = df[['price']]
lm = LinearRegression()
lm
lm.fit(X,Y)
lm.score(X, Y)

##############
# Question 6 #
##############
# fitting model to predict price using sqft_living
X = df[['sqft_living']]
Y = df[['price']]
lm = LinearRegression()
lm
lm.fit(X,Y)

# calculating R^2 of fit
lm.score(X, Y)

##############
# Question 7 #
##############
# creating a list of predictors
features = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement", "view",
            "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living"]
X = df[features]

# fitting model using list of features
lm.fit(X, Y)

# predicting model using fit
Yhat = lm.predict(X)
Yhat[0:5]
lm.score(X, Y)


# creating a list of tuples
Input=[('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model', LinearRegression())]

##############
# Question 8 #
##############
# using list of tuples to create a pipeline object
pipe=Pipeline(Input)
pipe

# fitting model
pipe.fit(X, Y)

# scoring the regression model
pipe.score(X, Y)

#############################################
# Module 5: MODEL EVALUATION AND REFINEMENT #
#############################################
# importing libraries
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
# print("done")

# setting up traing test split
features = ["floors", "waterfront", "lat", "bedrooms", "sqft_basement", "view",
            "bathrooms", "sqft_living15", "sqft_above", "grade", "sqft_living"]

X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)

# print("number of test samples :", x_test.shape[0])
# print("number of training samples :", x_train.shape[0])

##############
# Question 9 #
##############
# importing libraries
from sklearn.linear_model import Ridge

# creating a ridge object
RidgeModel=Ridge(alpha=0.1)

# fitting model using training data
RidgeModel.fit(x_train, y_train)

# calculating R^2 for training model
print(RidgeModel.score(x_train, y_train))

###############
# Question 10 #
###############
# creating a second order polynomial transformation
pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[features])
x_test_pr=pr.fit_transform(x_test[features])

# creating a separate ridge reg for polynomial transformation
RidgeModel=Ridge(alpha=0.1)

# fitting the model
RidgeModel.fit(x_train_pr, y_train)

# scoring polynomial model
print(RidgeModel.score(x_train_pr, y_train))




























# in order to display plot within window
# plt.show()
