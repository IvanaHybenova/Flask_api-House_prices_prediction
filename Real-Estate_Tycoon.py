# -*- coding: utf-8 -*-
"""
Real-Estate_Tycoon

@author: Ivana Hybenoa
"""

"""
1. E X P L O R A T O R Y   A N A L Y S I S
"""

# NumPy for numerical computing
import numpy as np

# Pandas for DataFrames
import pandas as pd
pd.set_option('display.max_columns', 100)

# Matplotlib for visualization
from matplotlib import pyplot as plt
# display plots in the notebook
%matplotlib inline 

# Seaborn for easier visualization
import seaborn as sns

# Load real estate data from CSV
df = pd.read_csv('real_estate_data.csv')

# Split X and y into train and test sets
# Function for splitting training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns = 'tx_price'),
                                                    df['tx_price'],
                                                    test_size = 0.2,
                                                    random_state = 0)

# Save test set as raw_data to test the app
df.to_csv('X_test.csv', index=None)

# Dataframe dimensions
df.shape

# Column datatypes
df.dtypes

# Display first 5 rows of df
df.head()

# Filter and display only df.dtypes that are 'object'
df.dtypes[df.dtypes == "object"]

# Loop through categorical feature names and print each one
for feature in df.dtypes[df.dtypes == "object"].index:
    print(feature)
    
# Display the first 10 rows of data
df.head(10)

# Display last 5 rows of data
df.tail(5)

# Plot histogram grid
df.hist(figsize = (14,14), xrot = -45)
# Clear the text "residue"
plt.show()

# Summarize numerical features
df.describe()

# Summarize categorical features
df.describe(include = "object")

# Plot bar plot for each categorical feature
for feature in df.dtypes[df.dtypes == 'object'].index:
    sns.countplot(y=feature, data=df)
    plt.show()


# SEGMENTATIONS ===========================================

# Segment tx_price by property_type and plot distributions
sns.boxplot(y='property_type', x='tx_price', data=df)

# Segment by property_type and display the means within each class
df.groupby('property_type').mean()

# On average, it looks like single family homes are more expensive.
# How else do the different property types differ? Let's see:
# Segment sqft by sqft and property_type distributions
sns.boxplot(y='property_type', x='sqft', data=df)

# Segment by property_type and display the means and standard deviations within each class
df.groupby('property_type').agg(['mean', 'std'])


# CORRELATIONS ======================================================
# Calculate correlations between numeric features
correlations = df.corr()

#Visualize the correlation grid with a heatmap to make it easier to digest.
# Make the figsize 7 x 6
plt.figure(figsize=(7,6))

# Plot heatmap of correlations
sns.heatmap(correlations)

# Change color scheme
sns.set_style("white")

# Make the figsize 10 x 8
plt.figure(figsize=(10,8))

# Plot heatmap of correlations
sns.heatmap(correlations)

# Make the figsize 10 x 8
plt.figure(figsize=(10, 8))

# Plot heatmap of annotated correlations
sns.heatmap(correlations * 100, annot=True, fmt='.0f')

# Generate a mask for the upper triangle
mask = np.zeros_like(correlations, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Make the figsize 10 x 8
plt.figure(figsize=(10,8))

# Plot heatmap of correlations
sns.heatmap(correlations * 100, annot=True, fmt='.0f', mask=mask)

# Make the figsize 9 x 8
plt.figure(figsize=(9,8))

# Plot heatmap of correlations
sns.heatmap(correlations * 100, annot=True, fmt='.0f', mask=mask, cbar=False)

'''
2. D A T A   C L E A N I N G
'''
import pandas as pd
pd.set_option('display.max_columns', 100)

import numpy as np

from matplotlib import pyplot as plt
%matplotlib inline

import seaborn as sns

# 1. Drop unwanted observations==========
# Drop duplicates
df = df.drop_duplicates()
print( df.shape )

# Fix structural errors =============
# Display unique values of 'basement'
df.basement.unique()

# Missing basement values should be 0
df['basement'] = df.basement.fillna(0)

# Display unique values of 'basement'
df.basement.unique()

# Next, to check for typos or inconsistent capitalization, display all the class distributions for the 'roof' feature.
# Class distributions for 'roof'
sns.countplot(y='roof', data=df)

# 'composition' should be 'Composition'
df.roof.replace('composition', 'Composition', inplace=True)

# 'asphalt' should be 'Asphalt'
df.roof.replace('asphalt', 'Asphalt', inplace=True)

# 'shake-shingle' and 'asphalt,shake-shingle' should be 'Shake Shingle'
df.roof.replace(['shake-shingle', 'asphalt,shake-shingle'], 'Shake Shingle', inplace=True)

# Class distributions for 'exterior_walls'
sns.countplot(y='exterior_walls', data=df)

# 'Rock, Stone' should be 'Masonry'
df.exterior_walls.replace('Rock, Stone', 'Masonry', inplace=True)

# 'Concrete' and 'Block' should be 'Concrete Block'
df.exterior_walls.replace(['Concrete', 'Block'], 'Concrete Block', inplace=True)

# Class distributions for 'exterior_walls'
sns.countplot(y='exterior_walls', data=df)

# 3. Remove unwanted outliers ========================================
# Box plot of 'tx_price' using the Seaborn library
sns.boxplot(df.tx_price)
plt.show()

# Violin plot of 'tx_price' using the Seaborn library
sns.violinplot(df.tx_price)
plt.show()
# Based on the violin plot for 'tx_price', it doesn't look like anything really stands out as a possible outlier.
# Violin plot of beds
sns.violinplot(df.beds)
plt.show()

# Violin plot of sqft
sns.violinplot(df.sqft)
plt.show()

# Violin plot of lot_size
sns.violinplot(df.lot_size)
plt.show()

# Among those three features, it looks like lot_size has a potential outlier!
# Look at its long and skinny tail.
# Let's look at the largest 5 lot sizes just to confirm.
# Sort df.lot_size and display the top 5 samples
df.lot_size.sort_values(ascending=False).head()
# Remove lot_size outliers
df = df[df.lot_size <= 500000]
# print length of df
print( len(df) )

# 4. Label missing categorical data =======================================
# Display number of missing values by feature (categorical)
df.select_dtypes(include=['object']).isnull().sum()

# Fill missing categorical values
for column in df.select_dtypes(include=['object']):
    df[column] = df[column].fillna('Missing')
    
# Display number of missing values by feature (categorical)
df.select_dtypes(include=['object']).isnull().sum()

# 5. Flag and fill missing numeric data
# Display number of missing values by feature (numeric)
df.select_dtypes(exclude=['object']).isnull().sum()


# 6. Save the cleaned dataframe
# Save cleaned dataframe to new file
df.to_csv('cleaned_df.csv', index=None)

'''
3. F E A T U R E   E N G I N E E R I N G
'''
# Now, let's display the first 5 rows from the dataset, just so we can have all of the existing features in front of us.
df.head()

# 1. Domain knowledge ============================
# Try to think of specific information you might want to isolate.

#For example, let's say you knew that homes with 2 bedrooms and 2 bathrooms are especially popular for investors.

# Maybe you suspect these types of properties command premium prices. (You don't need to know for sure.)
# Sure, number of bedrooms and number of bathrooms both already exist as features in the dataset.
# However, they do not specifically isolate this type of property.
# Therefore, you could create an indicator variable just for properties with 2 beds and 2 baths.
# Create indicator variable for properties with 2 beds and 2 baths
df['two_and_two'] = ((df.beds == 2) & (df.baths == 2)).astype(int)  # .astype(int) so we get 1 and 0

# Display percent of rows where two_and_two == 1
df.two_and_two.mean()

# Next, let's consider the housing market recession.
# According to data from Zillow, the lowest housing prices were from 2010 to end of 2013 (country-wide).
# Create indicator feature for transactions between 2010 and 2013, inclusive
df['during_recession'] = ((df.tx_year >= 2010) & (df.tx_year <= 2013)).astype(int)
# df['during_recession'] = df.tx_year.between(2010, 2013).astype(int)

# Print percent of transactions where during_recession == 1
print( df.during_recession.mean() )

# 2. Interaction features =====================================================
# For example, in our dataset, we know the transaction year and the year the property was built in.
# However, the more useful piece of information that combining these two features provides is the age 
# of the property at the time of the transaction.
# Create a property age feature
df['property_age'] = df.tx_year - df.year_built

# Sanity check
# It's always nice to do a quick sanity check after creating a feature, which could save you headaches down the road.
# For example, 'property_age' should never be less than 0, right?
print( df.property_age.min() )  # -8
# Number of observations with 'property_age' < 0
print( sum(df.property_age < 0) )  # 19   - possibly bought before construction

# However, for this problem, we are only interested in houses that already exist because the REIT only buys existing ones!
df = df[df.property_age >= 0]
len(df)

# How about the number of quality schools nearby?
# Create a school score feature that num_schools * median_school
df['school_score'] = df.num_schools * df.median_school

# Display median school score
df.school_score.median()

# 3. Group spare classes =========================================
# Bar plot for exterior_walls
sns.countplot(y='exterior_walls', data = df)

# Group 'Wood Siding' and 'Wood Shingle' with 'Wood'
df.exterior_walls.replace(['Wood Siding', 'Wood Shingle'], 'Wood', inplace = True )

# List of classes to group
other_exterior_walls = ['Concrete Block', 'Stucco', 'Masonry', 'Other', 'Asbestos shingle']
# Group other classes into 'Other'
df.exterior_walls.replace(other_exterior_walls, 'Other', inplace = True)
# Bar plot for exterior_walls
sns.countplot(y = 'exterior_walls', data = df)

# Bar plot for roof
sns.countplot(y = 'roof', data = df)
df.roof.replace(['Composition', 'Wood Shake/ Shingles'], 'Composition Shingle', inplace = True)
# List of classes to group
other_roof = ['Other', 'Gravel/Rock', 'Roll Composition', 'Slate', 'Built-up', 'Asbestos', 'Metal']
# Group other classes into 'Other'
df.roof.replace(other_roof, 'Other', inplace = True)

# Bar plot for roof
sns.countplot(y = 'roof', data = df)

# 4. Encode dummy variables ============================================
# Create new dataframe with dummy features
df = pd.get_dummies(df, columns = ['exterior_walls', 'roof', 'property_type'], drop_first = True)
df.head()

# 5.Remove unused or redundant features
df.dtypes
# we don't have, but might be ID columns, Other text descriptions...

# Removing 'tx_year' could also be a good idea because we don't want our model being overfit to the transaction year.
# Since we'll be applying it to future transactions, we might want the algorithms to focus on learning patterns from the other features.
# Drop 'tx_year' and 'year_built' from the dataset
df = df.drop(['tx_year', 'year_built'], axis = 1) # axis=1, because we are dropping columns

# Save analytical base table
df.to_csv('clean_data.csv', index=None)

'''
4. M O D E L   T R A I N I N G
'''
# NumPy for numerical computing
import numpy as np

# Pandas for DataFrames
import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Matplotlib for visualization
from matplotlib import pyplot as plt
# display plots in the notebook
%matplotlib inline 

# Seaborn for easier visualization
import seaborn as sns

# Scikit-Learn for Modeling
import sklearn

# Import Elastic Net, Ridge Regression, and Lasso Regression
from sklearn.linear_model import ElasticNet, Ridge, Lasso

# Import Random Forest and Gradient Boosted Trees
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

df = pd.read_csv('clean_data.csv')
df.shape

# 1. Splitting the data
# Function for splitting training and test set
from sklearn.model_selection import train_test_split 

# Create separate object for target variable
y = df.tx_price

# Create separate object for input features
X = df.drop('tx_price', axis=1) #  axis = 1 because we are splitting columns

# Split X and y into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=1234)
print( len(X_train), len(X_test), len(y_train), len(y_test) )

# Data preprocessing pipeline ========================================
# Function for creating model pipelines
from sklearn.pipeline import make_pipeline

# For standardization
from sklearn.preprocessing import StandardScaler

make_pipeline(StandardScaler(), Lasso(random_state=123)) # example of model pipeline for Lasso regression

# Create pipelines dictionary 
pipelines = {
    'lasso' : make_pipeline(StandardScaler(), Lasso(random_state=123)),
    'ridge' : make_pipeline(StandardScaler(), Ridge(random_state=123))
}
# Add a pipeline for 'enet'
pipelines['enet'] = make_pipeline(StandardScaler(), ElasticNet(random_state=123))
# Add a pipeline for 'rf'
pipelines['rf'] = make_pipeline(StandardScaler(), RandomForestRegressor(random_state = 123))
# Add a pipeline for 'gb'
pipelines['gb'] = make_pipeline(StandardScaler(), GradientBoostingRegressor(random_state = 123))

# Check that we have all 5 algorithms, and that they are all pipelines
for key, value in pipelines.items():
    print( key, type(value) )

# 2. Declare hyperparameters to tune ==========================================

# List tuneable hyperparameters of our Lasso pipeline
pipelines['lasso'].get_params()

# Lasso hyperparameters
lasso_hyperparameters = { 
    'lasso__alpha' : [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10] 
}

# Ridge hyperparameters
ridge_hyperparameters = { 
    'ridge__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]  
}

# Elastic Net hyperparameters
# l1_ratio is the ratio of L1 penalty to L2 penalty.
# The default value is 0.5.
# When l1_ratio=1, it is Lasso regression
# When l1_ratio=0, it is Ridge regression
# However, we'll use the special algorithms for Ridge and Lasso, so let's try values between 0.1 and 0.9, in increments of 0.2.
enet_hyperparameters = { 
    'elasticnet__alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],                        
    'elasticnet__l1_ratio' : [0.1, 0.3, 0.5, 0.7, 0.9]  
}
# Random forest hyperparameters
# The second one we'll tune is max_features.
# This controls the number of features each tree is allowed to choose from.
# It's what allows your random forest to perform feature selection.
# The default value is 'auto', which sets max_features = n_features.
# Let's also try 'sqrt', which sets max_features = sqrt(n_features)
# And 0.33, which sets max_features = 0.33 * n_features
rf_hyperparameters = { 
    'randomforestregressor__n_estimators' : [100, 200],
    'randomforestregressor__max_features': ['auto', 'sqrt', 0.33],
}

# Boosted tree hyperparameters
# The second one we'll tune is called learning_rate.
# This shrinks the contribution of each tree.
# There is tradeoff between learning rate and number of trees.
# The default value is 0.1.
# We'll try 0.05, 0.1, and 0.2.

# Finally, we'll tune max_depth.
# This controls the maximum depth of each tree.
# The default value is 3.
# We'll try 1, 3, and 5.
gb_hyperparameters = { 
    'gradientboostingregressor__n_estimators': [100, 200],
    'gradientboostingregressor__learning_rate' : [0.05, 0.1, 0.2],
    'gradientboostingregressor__max_depth': [1, 3, 5]
}

# Create hyperparameters dictionary
hyperparameters = {
    'rf' : rf_hyperparameters,
    'gb' : gb_hyperparameters,
    'lasso' : lasso_hyperparameters,
    'ridge' : ridge_hyperparameters,
    'enet' : enet_hyperparameters
}

for key in ['enet', 'gb', 'ridge', 'rf', 'lasso']:
    if key in hyperparameters:
        if type(hyperparameters[key]) is dict:
            print( key, 'was found in hyperparameters, and it is a grid.' )
        else:
            print( key, 'was found in hyperparameters, but it is not a grid.' )
    else:
        print( key, 'was not found in hyperparameters')

# 3. Fit and tune models with cross-validation ================================
# Helper for cross-validation
from sklearn.model_selection import GridSearchCV
#
## Fitting and tuning a single model
#
## Create cross-validation object from Lasso pipeline and Lasso hyperparameters
#model = GridSearchCV(pipelines['lasso'], hyperparameters['lasso'], cv=10, n_jobs=-1)
## Fit and tune model
#model.fit(X_train, y_train)

# Fitting and tuning all the models through a loop
# Create empty dictionary called fitted_models
fitted_models = {}

# Loop through model pipelines, tuning each one and saving it to fitted_models
for name, pipeline in pipelines.items():
    # Create cross-validation object from pipeline and hyperparameters
    model = GridSearchCV(pipeline, hyperparameters[name], cv=10, n_jobs=-1, verbose = 8)
    
    # Fit model on X_train, y_train
    model.fit(X_train, y_train)
    
    # Store model in fitted_models[name] 
    fitted_models[name] = model
    
    # Print '{name} has been fitted'
    print(name, 'has been fitted.')
    
# Check that we have 5 cross-validation objects
for key, value in fitted_models.items():
    print( key, type(value) )

# check that the models have been fitted correctly.
from sklearn.exceptions import NotFittedError

for name, model in fitted_models.items():
    try:
        pred = model.predict(X_test)
        print(name, 'has been fitted.')
    except NotFittedError as e:
        print(repr(e))
        
# 4. Evaluate models and select the winner ====================================
# display the best_score_ attribute for each fitted model
# for regression is best_score the average R^2 on the holdout folds
from sklearn.metrics import r2_score

for name, model in fitted_models.items():
    print( name, model.best_score_ )

# Mean absolute error (MAE)
# Another metric that would be especially useful for this problem is mean absolute error, or MAE.
# Remember, our win-condition for this project is predicting within $70,000 of true transaction prices, on average.
# Mean absolute error (or MAE) is the average absolute difference between predicted and actual values for our target variable. That exactly aligns with the terms of our win condition!
from sklearn.metrics import mean_absolute_error

## Example of predicting on a single model
## Predict test set using fitted random forest
#pred = fitted_models['rf'].predict(X_test)
## Calculate and print R^2 and MAE
#print( 'R^2:', r2_score(y_test, pred ))
#print( 'MAE:', mean_absolute_error(y_test, pred))

# Use a for loop, print the performance of each model in fitted_models on the test set.
for name, model in fitted_models.items():
    pred = model.predict(X_test)
    print( name )
    print( '--------' )
    print( 'R^2:', r2_score(y_test, pred ))
    print( 'MAE:', mean_absolute_error(y_test, pred))
    print()
    
# 5. Next, ask yourself these questions to pick the winning model: ==============

# Which model had the highest R2 on the test set?
# Random forest

# Which model had the lowest mean absolute error?
# Random forest

# Are these two models the same one?
# Yes

# Did it also have the best holdout R2 score from cross-validation?
# No

# Does it satisfy our win condition?
# Yes, its mean absolute error is less than $70,000!

# Finally, let's plot the performance of the winning model on the test set.

# It first plots a scatter plot.
# Then, it plots predicted transaction price on the X-axis.
# Finally, it plots actual transaction price on the y-axis.
rf_pred = fitted_models['rf'].predict(X_test)
plt.scatter(rf_pred, y_test)
plt.xlabel('predicted')
plt.ylabel('actual')
plt.show()

# This last visual check is a nice way to confirm our model's performance.
# Are the points scattered around the 45 degree diagonal?

# 6. Saving the winning model ==================================================
# First, let's take a look at the data type of your winning model.
type(fitted_models['rf'])

# we can use the best_estimator_ method to access it:
type(fitted_models['rf'].best_estimator_)

# If we output that object directly, we can also see the winning values for our hyperparameters.
fitted_models['rf'].best_estimator_

# The winning values for our hyperparameters are:
# n_estimators: 200
# max_features : 'auto'

# Great, now let's import a helpful package called pickle, which saves Python objects to disk.
import pickle

with open('model/final_model.pkl', 'wb') as f:
    pickle.dump(fitted_models['rf'].best_estimator_, f)

