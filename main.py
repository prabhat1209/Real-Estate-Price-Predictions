# Real Estate - Price Prediction

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load

# Reading Data

housing = pd.read_csv('data.csv')

#print(housing.head())
#print(housing.info())
#print(housing.describe())

#housing.hist(bins=50, figsize=(20, 15))
#plt.show()

# Splitting Train-Test Data

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
#print("Train : "+str(len(train_set))+"  &  Test : "+str(len(test_set)))

# Shuffling the Data

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

#print("Stt : "+str(strat_test_set['CHAS'].value_counts()))
#print("Stn : "+str(strat_train_set['CHAS'].value_counts()))

# Correlating the Data

housing = strat_train_set.copy()

corr_matrix = housing.corr()
#print(corr_matrix['MEDV'].sort_values(ascending=False))

# Visualizing the correlation

attributes = ["MEDV","RM","ZN","LSTAT"]
#scatter_matrix(housing[attributes], figsize=(12,8))
#plt.show()

# Trying out Attributes Combinations
#housing['TAXRM'] = housing['TAX']/housing['RM']

# Creating Pipeline

my_pipeline = Pipeline([ 
    ('imputer',SimpleImputer(strategy='median')),
    ('std_scaler',StandardScaler()),
])

housing = strat_train_set.drop('MEDV', axis=1)
housing_labels = strat_train_set['MEDV'].copy()

housing_num_tr = my_pipeline.fit_transform(housing)
#print(housing_num_tr)
#print(housing_num_tr.shape)

# Selecting a desired model for Real Estates Price Predictions

model_linear = LinearRegression()
model_linear.fit(housing_num_tr,housing_labels)
housing_linear_predictions = model_linear.predict(housing_num_tr)
lin_mse = mean_squared_error(housing_labels,housing_linear_predictions)
#print("Linear Regression : ",np.sqrt(lin_mse))

model_tree = DecisionTreeRegressor()
model_tree.fit(housing_num_tr,housing_labels)
housing_tree_predictions = model_tree.predict(housing_num_tr)
tree_mse = mean_squared_error(housing_labels,housing_tree_predictions)
#print("Decision Tree : ",np.sqrt(tree_mse))

# Using Better Evaluation Technique - Cross Validation 

scores = cross_val_score(model_tree, housing_num_tr, housing_labels, scoring='neg_mean_squared_error',cv=10)
rmse_scores = np.sqrt(-scores)
#print("Cross Validation : ",rmse_scores)

model_randomforest = RandomForestRegressor()
model_randomforest.fit(housing_num_tr,housing_labels)
housing_forest_predictions = model_randomforest.predict(housing_num_tr)
forest_mse = mean_squared_error(housing_labels,housing_forest_predictions)
#print("RandomForestRegressor : ",np.sqrt(forest_mse))

# Extracting the Model

dump(model_randomforest,'Real_Estate.joblib')

X_test = strat_test_set.drop('MEDV',axis=1)
Y_test = strat_test_set['MEDV'].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model_randomforest.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test,final_predictions)
#print('Final Error : ',np.sqrt(final_mse))

final_model = load('Real_Estate.joblib')
print(final_model.predict([[5,4,1,1,1,11,49,7,26,1,1,1,66]]))