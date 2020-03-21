#####################################
# Week 3: Exploratory Data Analysis #
#####################################
# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#######################
# Reading the Dataset #
#######################
# read url
path = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv"
df = pd.read_csv(path)
df.head()

# listing data type
df.dtypes

##############
# Question 1 #
##############
# The data type of column peak-rpm is float type

# the following prints a correlation matrix
df.corr()

##############
# Question 2 #
##############
# now printing corr matrix for select columns of interest
df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr()

##################################
# Continuous numerical variables #
##################################

# Positive linear relationship
# using engine size as a predictor of price
#sns.regplot(x="engine-size", y="price", data=df)
#plt.ylim(0,)
# plt.show()
# obtaining metrics on fit, such as strenght of correlation
df[["engine-size", "price"]].corr()

# Negative linear relationship
# likewise regressing price on highway mpg
#sns.regplot(x="highway-mpg", y="price", data=df)
# plt.show()
df[['highway-mpg', 'price']].corr()
# here we see a negative correlation

# Weak Linear Relationship
#sns.regplot(x="peak-rpm", y="price", data=df)
# plt.show()
df[['peak-rpm', 'price']].corr()
# weak inconclusive relation

#################
# Question 3 a) #
#################
df[['stroke', 'price']].corr()
# the correlation between stroke and price is 0.08
# on the weaker side

#################
# Question 3 b) #
#################
# given the above results, we can expect a weak linear positive result
#sns.regplot(x="stroke", y="price", data=df)
#plt.show()
# as expected, we find a weak linear relation

#########################
# Categorical variables #
#########################
# exploring the boxplot
# price vs body style boxplot
# sns.boxplot(x="body-style", y="price", data=df)
# plt.show()

# price vs drive wheels boxplot
# sns.boxplot(x="drive-wheels", y="price", data=df)
# plt.show()

####################################
# Descriptive Statistical Analysis #
####################################
# running some descriptive stats
df.describe

# printing some descriptives
df.describe(include=['object'])

################
# Value Counts #
################
# basic frequency tabs
df['drive-wheels'].value_counts()

# writing frequency tabs to dataframe
df['drive-wheels'].value_counts().to_frame()

# saving the written dataframe explicitly
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
# renaming columns for convenience
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts

# renaming index
drive_wheels_counts.index.name = 'drive_wheels'
drive_wheels_counts

# similarly for engine-location we have
engine_loc_counts = df['engine-location'].value_counts().to_frame()
# renaming columns for clarity
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
# renaming index
engine_loc_counts.index.name = 'engine-location'
engine_loc_counts.head(10)

######################
# Basics of Grouping #
######################
# obtaining distinct values
df['drive-wheels'].unique()

# creating a subset of dataframe
df_group_one = df[['drive-wheels', 'body-style', 'price']]
df_group_one

# grouping results
df_group_one = df_group_one.groupby(['drive-wheels'], as_index=False).mean()
df_group_one

# grouping by multiple variables
df_gptest = df[['drive-wheels', 'body-style', 'price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'], as_index=False).mean()
grouped_test1

# creating a pivot table of means
grouped_pivot = grouped_test1.pivot(index='drive-wheels', columns='body-style')
grouped_pivot

# filling missing cells in pivot with 0's
grouped_pivot = grouped_pivot.fillna(0)
grouped_pivot

##############
# Question 4 #
##############
# creating subset
df_group_2 = df[['drive-wheels', 'body-style', 'price']]

# grouping by body style to determine price
df_group_2 = df_group_2.groupby(['body-style'], as_index=False).mean()
df_group_2

# Variables: Drive Wheels and Body Style vs Price

# using the grouped results
#plt.pcolor(grouped_pivot, cmap='RdBu')
#plt.colorbar()
#plt.show()

# now setting up intricacies of our plot
fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

# label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

# centering labels
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

# insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(row_labels, minor=False)

# rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
# plt.show()

#############################
# Correlation and Causation #
#############################
# basic overall correlation
df.corr()

# Wheel-base vs Price
# calculating pearson correlation coefficient
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
# print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

# Horsepower vs Price
pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
# print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

# Length vs Price
pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
# print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

# Width vs Price
pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
# print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

# Curb-weight vs Price
pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
# print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

# Engine-size vs Price
pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
# print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

# Bore vs Price
pearson_coef, p_value = stats.pearsonr(df['bore'], df['price'])
# print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

# City-mpg vs Price
pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
# print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

# Highway-mpg vs Price
pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['price'])
# print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)

#########
# ANOVA #
#########
# ANOVA: Analysis of Variance

# Drive wheels
# anova compares same variable for different groups

grouped_test2=df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])
grouped_test2.head(2)

df_gptest

# getting group averaged values for 4wd
grouped_test2.get_group('4wd')['price']

# ANOVA testing across drive wheel types
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'], grouped_test2.get_group('4wd')['price'])
# print("ANOVA results: F=", f_val, ", P=", p_val)

# conducting ANOVA for two groups at a time: fwd and rwd
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'])
# print("ANOVA results: F=", f_val, ", P=", p_val)

# conducting ANOVA for: 4wd and rwd
f_val, p_val = stats.f_oneway(grouped_test2.get_group('rwd')['price'], grouped_test2.get_group('4wd')['price'])
# print("ANOVA results: F=", f_val, ", P=", p_val)

# conducting ANOVA for: 4wd and fwd
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('4wd')['price'])
# print("ANOVA results: F=", f_val, ", P=", p_val)

# anova is still most significant for diff bw fwd and rwd, 3 way results are driven by that difference.




















# in order to display plot within window
# plt.show()
