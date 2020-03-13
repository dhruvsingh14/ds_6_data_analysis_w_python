#####################################
# Week 3: Exploratory Data Analysis #
#####################################
# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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





















# in order to display plot within window
# plt.show()
