#Decision Tree
 
# About the data: 
# Let’s consider a Company dataset with around 10 variables and 400 records. 
# The attributes are as follows: 
#  Sales -- Unit sales (in thousands) at each location
#  Competitor Price -- Price charged by competitor at each location
#  Income -- Community income level (in thousands of dollars)
#  Advertising -- Local advertising budget for company at each location (in thousands of dollars)
#  Population -- Population size in region (in thousands)
#  Price -- Price company charges for car seats at each site
#  Shelf Location at stores -- A factor with levels Bad, Good and Medium indicating the quality of the shelving location for the car seats at each site
#  Age -- Average age of the local population
#  Education -- Education level at each location
#  Urban -- A factor with levels No and Yes to indicate whether the store is in an urban or rural location
#  US -- A factor with levels No and Yes to indicate whether the store is in the US or not
# The company dataset looks like this: 
 
# Problem Statement:
# A cloth manufacturing company is interested to know about the segment or attributes causes high sale. 
# Approach - A decision tree can be built with target variable Sale (we will first convert it in categorical variable) & all other variable will be independent in the analysis.  

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Company_Data.csv')

# creating dummy columns for the categorical columns 
dataset.columns
dummies = pd.get_dummies(dataset[['ShelveLoc',  'Urban', 'US']])
# Dropping the columns for which we have created dummies
dataset.drop(['ShelveLoc',  'Urban', 'US'],inplace=True,axis = 1)

# adding the columns to the dataset data frame 
dataset = pd.concat([dataset,dummies],axis=1)

dataset.head(10)

# To get the count of null values in the data 

dataset.isnull().sum() #no na values
dataset.shape 

# spillting into X as input and y as output variables
X = dataset.iloc[:, 1: ]
y = dataset.iloc[:, [0]]

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X.iloc[:,0:7]=sc_X.fit_transform(X.iloc[:,0:7])


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = regressor.predict(X_test).reshape(-1,1)

rmse = np.sqrt(np.mean((y_pred - y_test)**2))
