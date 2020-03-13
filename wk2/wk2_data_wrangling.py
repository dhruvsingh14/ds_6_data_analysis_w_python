##########################
# Week 2: Data Wrangling #
##########################

# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#######################
# Reading the Dataset #
#######################
# read url
filename = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"

# create headers list
headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", "body-style",
        "drive-wheels","engine-location","wheel-base","length","width","height","curb-weight","engine-type",
        "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
        "peak-rpm","city-mpg","highway-mpg","price"]

# import data
df = pd.read_csv(filename, names = headers)

# print head
df.head()

######################################
# Identify and handle missing values #
######################################
# replace "?" to NaN - customizing missing values
df.replace("?", np.nan, inplace = True)
df.head(5)

# identifying missing values, creating df
missing_data = df.isnull()
missing_data.head(5)

# missing values count by column
#for column in missing_data.columns.values.tolist():
#    print(column)
#    print(missing_data[column].value_counts())
#    print("")

# dealing with missing data
# imputation using averages using 'normalized-losses' column

avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
#print("Average of normalized-losses:", avg_norm_loss)
# replacing nan by mean
df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace = True)

# calculate mean value for 'bore' column
avg_bore=df['bore'].astype('float').mean(axis=0)
#print("Average of bore:", avg_bore)
# replacing nan by mean
df["bore"].replace(np.nan, avg_bore, inplace=True)

##############
# Question 1 #
##############
# replacing nan in 'stroke' column:

# calculate mean value for 'stroke' column
avg_stroke = df["stroke"].astype("float").mean(axis=0)
#print("Average of stroke:", avg_stroke)
# replacing nan by mean
df["stroke"].replace(np.nan, avg_stroke, inplace = True)
# print(df["stroke"].head())

# doing the same for 'horsepower' and 'peak-rpm' columns

# calculate mean value for 'horsepower' column
avg_hp = df["horsepower"].astype("float").mean(axis=0)
#print("Average of horsepower:", avg_hp)
# replacing nan by mean
df["horsepower"].replace(np.nan, avg_hp, inplace = True)

# calculate mean value for 'peak-rpm' column
avg_peak_rpm = df["peak-rpm"].astype("float").mean(axis=0)
#print("Average of peak-rpm:", avg_peak_rpm)
# replacing nan by mean
df["peak-rpm"].replace(np.nan, avg_peak_rpm, inplace = True)

# frequency tab
df['num-of-doors'].value_counts()

# highest frequency returned
df['num-of-doors'].value_counts().idxmax()

# now imputing using mode
df["num-of-doors"].replace(np.nan, "four", inplace=True)

# dropping rows with missing values for price
df.dropna(subset=["price"], axis=0, inplace=True)

# re ordering index after dropped rows
df.reset_index(drop=True, inplace=True)
df.head()

#######################
# Correct data format #
#######################

# checking data types
df.dtypes

# converting data types
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")

# verifying
df.dtypes

########################
# Data Standardization #
########################
df.head()

# converting units: mpg to L/100km given conversion rate
df['city-L/100km'] = 235/df["city-mpg"]

# checking data
df.head()

##############
# Question 2 #
##############
# converting highway mpg to L/100km
df['highway-L/100km'] = 235/df["highway-mpg"]
df.head()

######################
# Data Normalization #
######################
# scaling all values by max value
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()

##############
# Question 3 #
##############
# normalizing height column
df['height'] = df['height']/df['height'].max()


###########
# Binning #
###########
# starting by converting type
df["horsepower"]=df["horsepower"].astype(int, copy=True)

# plotting histogram
plt.hist(df["horsepower"])

# set x/y labels and plot title
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")

# displayin plot
#plt.show()

# creating 3 equal width bins on hp, using min and max as endpoints
bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
bins

# setting group names
group_names = ['Low', 'Medium', 'High']

# using cut function to segment
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels = group_names, include_lowest=True)
df[['horsepower','horsepower-binned']].head(20)

# frequency tabs
df["horsepower-binned"].value_counts()

# now plotting binned data
plt.bar(group_names, df["horsepower-binned"].value_counts())

# setting labels
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")

# displaying plot
#plt.show()

######################
# Bins visualization #
######################
a = (0,1,2)

# draw histogram of attribute "horsepower" with bins 3
plt.hist(df["horsepower"], bins = 3)

# setting labels
plt.xlabel("horsepower")
plt.ylabel("count")
plt.title("horsepower bins")

# displaying plot
#plt.show()

##########################################
# Indicator variable (or dummy variable) #
##########################################
df.columns

# creating indicator vars for fuel type
dummy_variable_1 = pd.get_dummies(df["fuel-type"])
dummy_variable_1.head()

# changing colnames
dummy_variable_1.rename(columns={'gas':'fuel-type-diesel', 'diesel':'fuel-type-diesel'}, inplace=True)
dummy_variable_1.head()

# inserting dummy vars, with names changed into df
df = pd.concat([df, dummy_variable_1], axis=1)
# drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)

df.head()

##############
# Question 4 #
##############
# creating dummies for "aspiration"

# creating indicator vars for fuel type
dummy_variable_2 = pd.get_dummies(df["aspiration"])
dummy_variable_2.head()

# changing colnames
dummy_variable_2.rename(columns={'std':'aspiration-turbo', 'turbo':'aspiration-std'}, inplace=True)
dummy_variable_2.head()

##############
# Question 5 #
##############

# inserting dummy vars, with names changed into df
df = pd.concat([df, dummy_variable_2], axis=1)

# drop original column "aspiration" from "df"
df.drop("aspiration", axis = 1, inplace=True)

print(df.head())

# saving dataframe to a csv
df.to_csv('clean_df.csv')












# in order to display plot within window
# plt.show()
