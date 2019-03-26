import os
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import svm
from sklearn import ensemble
from sklearn.neural_network import MLPRegressor
from joblib import dump

default_path = r'C:\Users\Cyndi\Documents\Git Repository\Stat 404 Private\Project_Final'
folder_path = r'\Part1'
os.chdir(default_path)

DF = pd.read_csv('googleplaystore.csv')

# =============================================================================
# Data exploration and preprocessing:
# Remove duplicates and outliers, impute NA's,transform string data to numeric,
# bin numeric data to categorical, log transform numeric data to ensure normal distribution
# =============================================================================


# App: remove duplicated apps and keep the last occurance, some apps are scraped more than once
DF.drop_duplicates('App', keep='last', inplace=True)

# Rating: Remove outliers in Rating and impute missing Rating with average rating per cateogory
DF = DF[pd.isna(DF['Rating']) | ((DF['Rating'] <= 5) & (DF['Rating'] >= 1))]
DF.reset_index(drop=True, inplace=True)
# Create a new function that fill the na with mean of each group, that handles new data
DF["Rating"] = DF.groupby("Category").transform(lambda x: x.fillna(x.mean()))

# Reviews: Remove number of reviews since it more installs has more reviews,
# the variable links the future
DF = DF.drop('Reviews', axis=1)

# Size: Transform Size to int and transform it into categories due to existence
# of "Varies with device"
DF['Size'] = DF['Size'].apply(lambda x: x.replace('M', '000000').replace('.', '').replace('k', '000'))
#plt.hist(DF[DF['Size'] != 'Varies with device']['Size'].apply(lambda x: math.log10(int(x))), bins=10)
DF['size_category'] = ' '
for i in range(len(DF)):
    if DF.loc[i, 'Size'] == 'Varies with device':
        DF.loc[i, 'size_category'] = 'Varies with device'
    elif  int(DF.loc[i, 'Size']) < 10000000:
        DF.loc[i, 'size_category'] = 'Size < 10MB'
    elif  int(DF.loc[i, 'Size']) < 50000000:
        DF.loc[i, 'size_category'] = '10MB<=Size<50MB'
    else:
        DF.loc[i, 'size_category'] = 'Size >= 50MB'

# Installs: Transform installs to int， and transform to log scale for normality
def int_installs(x):
    return int(x.replace(',', '').replace('+', ''))

def log_transform(x):
    return math.log10(x+1)

DF['Installs'] = DF['Installs'].apply(int_installs)
DF['log_installs'] = DF['Installs'].apply(log_transform)

# Type: type is good
DF['Type'].value_counts()

# Price: transform price to float, data is extremely right skewed, but this the data,
# will need to further exploration
def float_price(x):
    return float(x.replace('$', ''))
DF['Price'] = DF['Price'].apply(float_price)
DF['log_Price'] = DF['Price'].apply(log_transform)

DF['Price'].value_counts(sort=False).sort_index()
DF['Price'].mean()
DF['Price'].std()

# Content Rating: combine 'Adults only 18+' to 'Mature 17+' and 'Unrated' to 'Everyone' due to lack of observation
DF['Content Rating'].value_counts()
DF['Content Rating'] = ['Mature 17+' if r == 'Adults only 18+' else 'Everyone' if r == 'Unrated' else r for r in DF['Content Rating']]

# Last Updated: transfer Last updated to date, and extract year and month
DF['Last Updated'] = pd.to_datetime(DF['Last Updated'])
DF['Year'] = [d.year for d in DF['Last Updated']]
DF['Month'] = [d.month for d in DF['Last Updated']]
DF['Year'].value_counts()
DF['Month'].value_counts().sort_index()

# Current version: current version is app specific, need to drop it
DF = DF.drop('Current Ver', axis=1)

# Android Version: Group Android Verion to 4 categories
DF.dropna(subset=['Android Ver'], inplace=True) #drop 2 missing android version
DF.reset_index(drop=True, inplace=True)
DF['Android Version Group'] = ' '
for i in range(len(DF)):
    if DF.loc[i, 'Android Ver'] == 'Varies with device':
        DF.loc[i, 'Android Version Group'] = 'Varies with device'
    elif  int(DF.loc[i, 'Android Ver'][0]) < 4:
        DF.loc[i, 'Android Version Group'] = 'Allow Older than Android Ver 4'
    elif  int(DF.loc[i, 'Android Ver'][0]) < 5:
        DF.loc[i, 'Android Version Group'] = 'Android Version 4 and up'
    else:
        DF.loc[i, 'Android Version Group'] = 'Android Version newer than 5'
DF['Android Version Group'].value_counts()

# Multiple genres exiisted, separate it into count of genres and first genre
DF['Genres_count'] = DF['Genres'].apply(lambda x: len(x.split(';')))
DF['First_Genres'] = DF['Genres'].str.split(';', expand=True)[0]
# DF['First_Genres'].value_counts()
# There are 48 different genres, too many to include for linear regression

# =============================================================================
# Data Visualization
# =============================================================================
#%matplotlib inline

plt.hist(DF['Price'], bins=100)
plt.hist(DF['log_Price'], bins=100)

AX = plt.gca()
DF['size_category'].value_counts().plot(kind='barh', color='0.75', x="Count", y="size_category", ax=AX)
AX.set_title("Count per Size Category")
AX.set_xlabel("App Count")
AX.set_ylabel("Size Category")
plt.show()

plt.hist(DF['Installs'], bins=10)
plt.hist(DF['log_installs'], bins=10)
plt.title('Histogram of Number of Installs')
plt.xlabel('Number of Intalls')
plt.ylabel('Number of Apps')
plt.show()

plt.hist(DF['log_installs'], bins=10)
plt.title('Histogram of Number of Installs in Log Scale')
plt.xlabel('Number of Intalls in Log base 10')
plt.ylabel('Number of Apps')
plt.show()

AX = plt.gca()
DF['Android Version Group'].value_counts().plot(kind='barh', color='0.75', ax=AX)
AX.set_title("Count per Android Version Group")
AX.set_xlabel("App Count")
AX.set_ylabel("Android Version Group")
plt.show()

# =============================================================================
# Prepare train and test data and result table
# =============================================================================

# 60% training data, 25% validation, 15% final testing
# Need a function to get dummies by using reindex
DF_X1 = DF[['Category', 'Type', 'Content Rating', 'Android Version Group', 'size_category', \
           'log_Price', 'Year', 'Month', 'Genres_count', 'First_Genres']]
DF_X1 = pd.get_dummies(DF_X1)
DF_X = pd.merge(DF_X1, DF['Rating'], left_index=True, right_index=True)
DF_Y = DF['log_installs']

X_TRAIN, X_VALID, Y_TRAIN, Y_VALID = train_test_split(DF_X, DF_Y, \
                                      test_size=0.40, \
                                      random_state=2019)

X_VALID, X_TEST, Y_VALID, Y_TEST = train_test_split(X_VALID, Y_VALID, \
                                      test_size=0.375, \
                                      random_state=2019)

SCORE_TABLE = pd.DataFrame(columns=['Model Name', 'Train Score', 'Validation Score'])
def ROW_COUNTER():
    def add():
        counter[0] = counter[0] + 1
        return counter
    counter = [0]
    return add
ROW_COUNT = ROW_COUNTER()

# =============================================================================
# Intial Model: multiple linear regression
# =============================================================================

M1 = linear_model.LinearRegression().fit(X_TRAIN, Y_TRAIN)
M1_TRAIN_SCORE = M1.score(X_TRAIN, Y_TRAIN) #R^2 = 0.289 on training data
#M1_prediction = M1.predict(X_VALID)
#plt.scatter(Y_VALID,M1_prediction)
M1_VALID_SCORE = M1.score(X_VALID, Y_VALID) #R^2 = 0.295 on validation data
SCORE_TABLE.loc[0] = ['Multiple linear regression', M1_TRAIN_SCORE, M1_VALID_SCORE]

# =============================================================================
# Model #2, ridge regression
# =============================================================================

C = [0.01, 0.1, 1, 10, 100]
for c in C:
    M2 = linear_model.Ridge(alpha=c).fit(X_TRAIN, Y_TRAIN)
    M2_TRAIN_SCORE = M2.score(X_TRAIN, Y_TRAIN)
    M2_VALID_SCORE = M2.score(X_VALID, Y_VALID)
    SCORE_TABLE .loc[ROW_COUNT()[0]] = [f'Ridge Regression with C = {c}', M2_TRAIN_SCORE, M2_VALID_SCORE]

# =============================================================================
# Model #3, lasso regression
# =============================================================================

for c in C:
    M3 = linear_model.Lasso(alpha=c).fit(X_TRAIN, Y_TRAIN)
    M3_TRAIN_SCORE = M3.score(X_TRAIN, Y_TRAIN)
    M3_VALID_SCORE = M3.score(X_VALID, Y_VALID)
    SCORE_TABLE.loc[ROW_COUNT()[0]] = [f'Lasso Regression with C = {c}', M3_TRAIN_SCORE, M3_VALID_SCORE]

# =============================================================================
# Model #4, SVM regression, SVR
# =============================================================================

for c in C:
    M4 = svm.SVR(C=c, gamma='scale').fit(X_TRAIN, Y_TRAIN)
    M4_TRAIN_SCORE = M4.score(X_TRAIN, Y_TRAIN)
    M4_VALID_SCORE = M4.score(X_VALID, Y_VALID)
    SCORE_TABLE .loc[ROW_COUNT()[0]] = [f'SVM Regression with C = {c}', M4_TRAIN_SCORE, M4_VALID_SCORE]

# =============================================================================
# Model #5, random forest
# =============================================================================

M5 = ensemble.RandomForestRegressor(n_estimators=100, min_samples_leaf=20, random_state=1992).fit(X_TRAIN, Y_TRAIN)
M5_TRAIN_SCORE = M5.score(X_TRAIN, Y_TRAIN)
M5_VALID_SCORE = M5.score(X_VALID, Y_VALID)
SCORE_TABLE .loc[ROW_COUNT()[0]] = [f'Random Forest', M5_TRAIN_SCORE, M5_VALID_SCORE]

# =============================================================================
# Model #6, gradient boosting tree
# =============================================================================

M6 = ensemble.GradientBoostingRegressor(n_estimators=100, min_samples_leaf=20, random_state=1992).fit(X_TRAIN, Y_TRAIN)
M6_TRAIN_SCORE = M6.score(X_TRAIN, Y_TRAIN)
M6_VALID_SCORE = M6.score(X_VALID, Y_VALID)
SCORE_TABLE .loc[ROW_COUNT()[0]] = [f'Gradient Boosting', M6_TRAIN_SCORE, M6_VALID_SCORE]

# =============================================================================
# Model #7, Adaboost
# =============================================================================

M7 = ensemble.AdaBoostRegressor(n_estimators=100, learning_rate=0.1, random_state=1992).fit(X_TRAIN, Y_TRAIN)
M7_TRAIN_SCORE = M7.score(X_TRAIN, Y_TRAIN)
M7_VALID_SCORE = M7.score(X_VALID, Y_VALID)
SCORE_TABLE .loc[ROW_COUNT()[0]] = [f'Adaboost', M7_TRAIN_SCORE, M7_VALID_SCORE]

# =============================================================================
# Model #8, Neural Network
# =============================================================================

M8 = MLPRegressor(hidden_layer_sizes=(100, 5)).fit(X_TRAIN, Y_TRAIN)
M8_TRAIN_SCORE = M8.score(X_TRAIN, Y_TRAIN)
M8_VALID_SCORE = M8.score(X_VALID, Y_VALID)
SCORE_TABLE .loc[ROW_COUNT()[0]] = [f'Neural Network', M8_TRAIN_SCORE, M8_VALID_SCORE]

SCORE_TABLE .to_csv('SCORE_TABLE .csv')

# =============================================================================
# Final testing with test data using Ridge regression and Gradient boost
# =============================================================================

print(f'The R^2 for gradient boost on test data is {round(M6.score(X_TEST, Y_TEST),4)}')

# =============================================================================
# Interpretion of gradient boost model
# =============================================================================

#need exponential to transform log_installs back to number of installs
FEATURE_IMPORTANCE = list(zip(X_TRAIN.columns, M6.feature_importances_))
FEATURE_IMPORTANCE = [i for i in FEATURE_IMPORTANCE if i[1]>0.01]
FEATURE = [i[0] for i in FEATURE_IMPORTANCE]
IMPORTANCE = [i[1] for i in FEATURE_IMPORTANCE]
plt.barh(FEATURE, IMPORTANCE)
plt.title("FEATURE IMPORTANCE")
#plt.xticks(rotation=90)
plt.show()

# =============================================================================
# Final model:
# =============================================================================

dump(M6, 'app_installations.joblib')
