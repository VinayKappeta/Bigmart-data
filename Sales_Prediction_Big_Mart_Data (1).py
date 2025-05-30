# Databricks notebook source
# %pip install xgboost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

train = pd.read_csv("/Workspace/Users/vinayreddy.kappeta@gilead.com/ML/train.csv")
test = pd.read_csv("/Workspace/Users/vinayreddy.kappeta@gilead.com/ML/test.csv")


# COMMAND ----------

train

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Exploration, EDA

# COMMAND ----------

train.shape

# COMMAND ----------

train.info()

# COMMAND ----------

train.describe()

# COMMAND ----------

#Item visibility is having zero which doesn't seems to be correct, so we need to explore on the zeros
zero_visibility_count = train[train.Item_Visibility == 0]
zero_visibility_count.shape

# COMMAND ----------

# MAGIC %md
# MAGIC 526 values are zero so we need to impute the values in the data cleaning procedure

# COMMAND ----------

for col in train.select_dtypes(include=['object']).columns:
    print(f"\nUnique values and counts for {col}:")
    print(train[col].value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC * Item_Identifier - So many unique values see if something can be extracted
# MAGIC * Item_Weight - Numeric , Missing value needs to be treated
# MAGIC * Item_Fat_Content - Convert to boolean
# MAGIC * Item_Visibility - Numeric # Treat the zeroes
# MAGIC * Item_Type - Categoricals
# MAGIC * Item_MRP - Numeric
# MAGIC * Outlet_Identifier - Categoricals
# MAGIC * Outlet_Establishment_Year - Can be dropped
# MAGIC * Outlet_Size - Can be Convert to Numeric ( 1,2,3) , fill missing values
# MAGIC * Outlet_Location_Type - Can be Convert to Numeric ( 1,2,3)
# MAGIC * Outlet_Type - Convert to Numeric ( 0, 1,2,3)

# COMMAND ----------

for col in train.iloc[:,0:len(train.columns)].columns:
    print(col,':',train[col].nunique(),':',train[col].isna().sum())
    #target_mean(train.col,'click_rate_log')

# COMMAND ----------

plt.figure(figsize=(12, 8))
for i, col in enumerate(train.select_dtypes(include=['float64', 'int64']).columns):
    plt.subplot(2, 3, i + 1)
    sns.histplot(train[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Observation: The above plots show that <br>
# MAGIC Item weight and Item_MRP is having a kind of normal distribution <br>
# MAGIC Item visibility and Item outlet sales are left skewed

# COMMAND ----------

plt.figure(figsize=(10, 8))
correlation_matrix = train.select_dtypes(include=['float64', 'int64']).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Observation: Based on the above plot the Item_outlet_sales and Item_MRP is having a kind of high correlation compared to the other column values

# COMMAND ----------

plt.scatter(train['Item_Visibility'],train['Item_Outlet_Sales'])
plt.xlabel("Item Visibility")
plt.ylabel("Item Outlet Sales")
plt.title("Item Visibility vs Item Outlet Sales")

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Items having visibility less than 0.2 sold them most

# COMMAND ----------

sns.barplot(x='Item_Type',y='Item_Outlet_Sales',data=train,saturation=15)
plt.xticks(rotation=90)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC starchy foods have most of the sales compared to other followed by seafood and Fruits and vegitables

# COMMAND ----------

sns.boxplot(x='Item_Type',y='Item_MRP',data=train,saturation=20)
plt.xticks(rotation=90)
plt.show()

# COMMAND ----------

sns.boxplot(x='Item_Type',y='Item_Outlet_Sales',data=train,saturation=20)
plt.xticks(rotation=90)
plt.show()

# COMMAND ----------

from scipy.stats import zscore

# Calculate Z-scores
train['sales_zscore'] = zscore(train['Item_Outlet_Sales'])
train['mrp_zscore'] = zscore(train['Item_MRP'])

# Identify outliers using Z-score (threshold = 3)
sales_outliers_zscore = train[(train['sales_zscore'].abs() > 3)]
mrp_outliers_zscore = train[(train['mrp_zscore'].abs() > 3)]

# Calculate IQR for sales
Q1_sales = train['Item_Outlet_Sales'].quantile(0.25)
Q3_sales = train['Item_Outlet_Sales'].quantile(0.75)
IQR_sales = Q3_sales - Q1_sales

# Identify outliers using IQR for sales
sales_outliers_iqr = train[(train['Item_Outlet_Sales'] < (Q1_sales - 1.5 * IQR_sales)) | (train['Item_Outlet_Sales'] > (Q3_sales + 1.5 * IQR_sales))]

# Calculate IQR for mrp
Q1_mrp = train['Item_MRP'].quantile(0.25)
Q3_mrp = train['Item_MRP'].quantile(0.75)
IQR_mrp = Q3_mrp - Q1_mrp

# Identify outliers using IQR for mrp
mrp_outliers_iqr = train[(train['Item_MRP'] < (Q1_mrp - 1.5 * IQR_mrp)) | (train['Item_MRP'] > (Q3_mrp + 1.5 * IQR_mrp))]

# Display outliers
display(len(sales_outliers_zscore))
display(len(mrp_outliers_zscore))
display(len(sales_outliers_iqr))
display(len(mrp_outliers_iqr))
train.drop(columns=['sales_zscore','mrp_zscore'],axis=1,inplace=True)

# COMMAND ----------

# MAGIC %md
# MAGIC As we can see the outliers are with only sales value, Going with Hypothesis we should not remove the outlier values because the sales might have been high

# COMMAND ----------

# MAGIC %md
# MAGIC #Data cleaning

# COMMAND ----------

null_data_frame = pd.DataFrame(train.isnull().sum().reset_index())
null_data_frame.columns = ['column_name','No_of_nulls']
null_data_frame['percentage_of_nulls'] = null_data_frame['No_of_nulls']*100/train.shape[0]
null_data_frame


# COMMAND ----------

null_data_frame = pd.DataFrame(test.isnull().sum().reset_index())
null_data_frame.columns = ['column_name','No_of_nulls']
null_data_frame['percentage_of_nulls'] = null_data_frame['No_of_nulls']*100/test.shape[0]
null_data_frame

# COMMAND ----------

# MAGIC %md
# MAGIC Based on observation 1 as the data is normally distributed for the column we can impute the Item_weight column with median/mean value grouping it by the Item.<br>
# MAGIC Based on observation 1 as the data is left skewed and the differnece between mean and median is less distributed for the column we can impute the Item_visbility column with mean value grouping it by the Item.

# COMMAND ----------

print("-------------------------train data-----------------------------------------------")
for item in train['Item_Identifier'].unique():
  print(item)
  print(train.loc[train['Item_Identifier'] == item, 'Item_Visibility'].mean())
print("-------------------------test data-----------------------------------------------")
for item in test['Item_Identifier'].unique():
  print(item)
  print(test.loc[train['Item_Identifier'] == item, 'Item_Visibility'].mean())

# COMMAND ----------

#as the visibility might differ from Item to Item so we need to impute with median, combination of both item identifier and item weight
print("==============Train data==========================================")
for item in train['Item_Identifier'].unique():
    train.loc[(train['Item_Identifier'] == item) & (train['Item_Visibility']==0), 'Item_Visibility'] = train.loc[train['Item_Identifier'] == item, 'Item_Visibility'].mean()
zero_visibility_count = train[train.Item_Visibility == 0]
print(zero_visibility_count.shape)
for item in train['Item_Identifier'].unique():
   mean_visibility = train.loc[train['Item_Identifier'] == item, 'Item_Visibility'].mean()
   if mean_visibility ==0:
       mean_visibility = train['Item_Visibility'].mean()
   train.loc[(train['Item_Identifier'] == item) & (train['Item_Visibility']==0), 'Item_Visibility'] = mean_visibility    
zero_visibility_count = train[train.Item_Visibility == 0]
print(zero_visibility_count.shape)
#as the visibility might differ from Item to Item so we need to impute with median, combination of both item identifier and item weight
print("==============Test data==========================================")
for item in test['Item_Identifier'].unique():
    test.loc[(test['Item_Identifier'] == item) & (test['Item_Visibility']==0), 'Item_Visibility'] = test.loc[test['Item_Identifier'] == item, 'Item_Visibility'].mean()
zero_visibility_count = test[test.Item_Visibility == 0]
print(zero_visibility_count.shape)
for item in test['Item_Identifier'].unique():
   mean_visibility = test.loc[test['Item_Identifier'] == item, 'Item_Visibility'].mean()
   if mean_visibility ==0:
       mean_visibility = test['Item_Visibility'].mean()
   test.loc[(test['Item_Identifier'] == item) & (test['Item_Visibility']==0), 'Item_Visibility'] = mean_visibility  
zero_visibility_count = test[test.Item_Visibility == 0]
print(zero_visibility_count.shape) 

# COMMAND ----------

print("-------------------------train data-----------------------------------------------")
for item in train['Item_Identifier'].unique():
  print(item)
  print(train.loc[train['Item_Identifier'] == item, 'Item_Weight'].mean())
print("-------------------------test data-----------------------------------------------")
for item in test['Item_Identifier'].unique():
  print(item)
  print(test.loc[train['Item_Identifier'] == item, 'Item_Weight'].mean())

# COMMAND ----------

#as the weight might differ from Item to Item so we need to impute with median/mean, combination of both item identifier and item weight
print("==============Train data==========================================")
for item in train['Item_Identifier'].unique():
    train.loc[(train['Item_Identifier'] == item) & (train['Item_Weight'].isnull()), 'Item_Weight'] = train.loc[train['Item_Identifier'] == item, 'Item_Weight'].mean()
zero_visibility_count = train['Item_Weight'].isna().sum()
print(zero_visibility_count.shape)
for item in train['Item_Identifier'].unique():
    mean_weight = train.loc[train['Item_Identifier'] == item, 'Item_Weight'].mean()
    if pd.isna(mean_weight):
       mean_weight = train['Item_Weight'].mean()
    train.loc[(train['Item_Identifier'] == item) & (train['Item_Weight'].isnull()), 'Item_Weight'] = mean_weight   
zero_visibility_count = train[train['Item_Weight'].isna()]
print(zero_visibility_count.shape)
#as the visibility might differ from Item to Item so we need to impute with median, combination of both item identifier and item weight
print("==============Test data==========================================")
for item in test['Item_Identifier'].unique():
    test.loc[(test['Item_Identifier'] == item) & (test['Item_Weight'].isnull()), 'Item_Weight'] = test.loc[test['Item_Identifier'] == item, 'Item_Weight'].mean()
zero_visibility_count = test['Item_Weight'].isna().sum()
print(zero_visibility_count.shape)
for item in test['Item_Identifier'].unique():
    mean_weight = test.loc[test['Item_Identifier'] == item, 'Item_Weight'].mean()
    if pd.isna(mean_weight):
       mean_weight = test['Item_Weight'].mean()
    test.loc[(test['Item_Identifier'] == item) & (test['Item_Weight'].isnull()), 'Item_Weight'] = mean_weight    
zero_visibility_count = test[test['Item_Weight'].isna()]
print(zero_visibility_count.shape) 

# COMMAND ----------

display(train.isnull().sum())
print("-----------------------------------------------")
display(test.isnull().sum())

# COMMAND ----------

for col in train.select_dtypes(include=['object']).columns:
    print(f"\nUnique values and counts for {col}:")
    print(train[col].value_counts())
print("---------------------------------------------------------")
for col in test.select_dtypes(include=['object']).columns:
    print(f"\nUnique values and counts for {col}:")
    print(test[col].value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Processsing and Feature Engineering

# COMMAND ----------

# MAGIC %md
# MAGIC As the data needs to be compared from the year 2013 so we need to check the age of store to analyse better than the year of establishment for sales relationship below cell is used for converting that
# MAGIC

# COMMAND ----------

train['Age_of_outlet'] = 2013-train['Outlet_Establishment_Year']
train.drop('Outlet_Establishment_Year',axis = 1, inplace=True)
test['Age_of_outlet'] = 2013-test['Outlet_Establishment_Year']
test.drop('Outlet_Establishment_Year',axis = 1, inplace=True)

# COMMAND ----------

test['Item_Fat_Content'] = test['Item_Fat_Content'].replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'})
display(test['Item_Fat_Content'].value_counts())
print("---------------------------------------------------------")
train['Item_Fat_Content'] = train['Item_Fat_Content'].replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'})
display(train['Item_Fat_Content'].value_counts())
print("---------------------------------------------------------")
train['Item_Fat_Content']=train['Item_Fat_Content'].replace({ 'Regular':1, 'Low Fat':0})
test['Item_Fat_Content']=test['Item_Fat_Content'].replace({ 'Regular':1, 'Low Fat':0})
display(test['Item_Fat_Content'].value_counts())
print("---------------------------------------------------------")
display(train['Item_Fat_Content'].value_counts())
print("---------------------------------------------------------")

# COMMAND ----------

display(train['Item_Identifier'].apply(lambda x: x[0:2]).value_counts())
display(test['Item_Identifier'].apply(lambda x: x[0:2]).value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC converting the item identifiers to proper for better model building and analysis

# COMMAND ----------

train['Item_Identifier'] = train['Item_Identifier'].apply(lambda x: x[0:2]).replace({'FD':'Food','NC':'Non-Consumable','DR':'Drinks'})
test['Item_Identifier'] = test['Item_Identifier'].apply(lambda x: x[0:2]).replace({'FD':'Food','NC':'Non-Consumable','DR':'Drinks'})

# COMMAND ----------

print(train['Item_Identifier'].value_counts())
print(test['Item_Identifier'].value_counts())

# COMMAND ----------

import matplotlib.pyplot as plt

# Group by Item_Identifier and sum the Item_Outlet_Sales
item_sales = train.groupby('Item_Identifier')['Item_Outlet_Sales'].sum()
item_sales

# COMMAND ----------

# MAGIC %md
# MAGIC If we observe food has more sales compared to other entities

# COMMAND ----------

# MAGIC %md
# MAGIC price to weight will give understanding on outlet sales as the chances are high for the sales to happen when price to weight is good

# COMMAND ----------

train['price/wt'] = train['Item_MRP'] /train['Item_Weight']
test['price/wt'] = test['Item_MRP'] /test['Item_Weight']

# COMMAND ----------

# MAGIC %md
# MAGIC Assuming Outlet size is dependent on the outlet type

# COMMAND ----------

# Display values of Outlet_Type with respect to Outlet_Size
outlet_type_size = train.groupby(['Outlet_Type', 'Outlet_Size']).size().reset_index(name='Count')
display(outlet_type_size)
train['Outlet_Size'] = train.apply(
    lambda x: 'Small' if pd.isnull(x['Outlet_Size']) and x['Outlet_Type'] == 'Grocery Store' else x['Outlet_Size'],
    axis=1
)
display(train['Outlet_Size'].value_counts())
# Display values of Outlet_Type with respect to Outlet_Size
outlet_type_size = test.groupby(['Outlet_Type', 'Outlet_Size']).size().reset_index(name='Count')
display(outlet_type_size)
test['Outlet_Size'] = test.apply(
    lambda x: 'Small' if pd.isnull(x['Outlet_Size']) and x['Outlet_Type'] == 'Grocery Store' else x['Outlet_Size'],
    axis=1
)
display(test['Outlet_Size'].value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC also outlet size is dependent on the location type as well

# COMMAND ----------

# Display values of Outlet_Type with respect to Outlet_Size
outlet_loc_type_size = train.groupby(['Outlet_Location_Type', 'Outlet_Size']).size().reset_index(name='Count')
display(outlet_loc_type_size)
train['Outlet_Size'] = train.apply(
    lambda x: 'Small' if pd.isnull(x['Outlet_Size']) and x['Outlet_Location_Type'] == 'Tier 2' else x['Outlet_Size'],
    axis=1
)
display(train['Outlet_Size'].value_counts())
# Display values of Outlet_Type with respect to Outlet_Size
outlet_loc_type_size = test.groupby(['Outlet_Location_Type', 'Outlet_Size']).size().reset_index(name='Count')
display(outlet_loc_type_size)
test['Outlet_Size'] = test.apply(
    lambda x: 'Small' if pd.isnull(x['Outlet_Size']) and x['Outlet_Location_Type'] == 'Tier 2' else x['Outlet_Size'],
    axis=1
)
display(test['Outlet_Size'].value_counts())

# COMMAND ----------

train['Item_MRP2'] =  np.where(train['Item_MRP'] <69,"A",
                              np.where(train['Item_MRP'] <136,"B",
                                       np.where(train['Item_MRP'] <203,"C","D")))
#train['Item_Visibility2'] =  np.where(train['Item_Visibility'] < 0.19,1,0)
test['Item_MRP2'] =  np.where(test['Item_MRP'] <69,"A",
                              np.where(test['Item_MRP'] <136,"B",
                                       np.where(test['Item_MRP'] <203,"C","D")))
#test['Item_Visibility2'] =  np.where(test['Item_Visibility'] < 0.19,1,0)

# COMMAND ----------

# Check co-relation with target for numeric columns
import seaborn as sns
import numpy as np
corr=train[['Item_Weight','Item_Visibility','Item_MRP','Age_of_outlet','price/wt','Item_Outlet_Sales']].corr()
mask=np.triu(np.ones_like(corr))
sns.heatmap(corr,annot=True,mask=mask,cbar=False)

# COMMAND ----------

df = train
df2 = test
numerical_cols = [col for col in df.columns if df[col].dtype != 'object']
categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
removing_encoding_list = ['Item_Identifier','Outlet_Identifier']
for val in removing_encoding_list:
  categorical_cols.remove(val)
print(numerical_cols)
print(categorical_cols)
print(df[categorical_cols])
numerical_cols = [col for col in df2.columns if df2[col].dtype != 'object']
categorical_cols = [col for col in df2.columns if df2[col].dtype == 'object']
removing_encoding_list = ['Item_Identifier','Outlet_Identifier']
for val in removing_encoding_list:
  categorical_cols.remove(val)
print(numerical_cols)
print(categorical_cols)
print(df2[categorical_cols])

# COMMAND ----------

# Create a label encoder object
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

# COMMAND ----------

# Fit and transform the categorical features
df['Outlet'] = encoder.fit_transform(df['Outlet_Identifier'])
for i in categorical_cols:
    df[i] = encoder.fit_transform(df[i])
df.head()
df2['Outlet'] = encoder.fit_transform(df2['Outlet_Identifier'])
for i in categorical_cols:
    df2[i] = encoder.fit_transform(df2[i])
df2.head()


# COMMAND ----------

data = pd.get_dummies(df, columns=['Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_MRP2'])
data2 = pd.get_dummies(df2, columns=['Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_MRP2'])

# COMMAND ----------

data.dtypes

# COMMAND ----------

df_numerical = data.select_dtypes(exclude=['object'])
pd.DataFrame(df_numerical).info()

# COMMAND ----------

data2.select_dtypes(exclude=['object']).info()

# COMMAND ----------

target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier','Outlet_Identifier']
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
def modelfit(model, train, predictors, target, IDcol):
    #Fit the algorithm on the data
    model.fit(train[predictors], train[target])
        
    #Predict training set:
    train_predictions = model.predict(train[predictors])

    #Perform cross-validation:
    cv_score = cross_val_score(model, train[predictors], train[target], cv=20, scoring='neg_mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    
    #Print model report:
    print("\nModel Report")
    print("RMSE : %.4g" % np.sqrt(mean_squared_error(train[target].values, train_predictions)))
    print("CV Score : Mean - %.4g | Std - %.4g | Min - %.4g | Max - %.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    
    #Predict on testing data:
    return model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Linear Regression

# COMMAND ----------

predictors = [x for x in data.columns if x not in [target]+IDcol]
# print predictors
linear = LinearRegression()
modelfit(linear, data, predictors, target, IDcol)

# COMMAND ----------

linear.feature_names_in_

# COMMAND ----------

linear_final = data2[['Item_Identifier','Outlet_Identifier']]
linear_pred = linear.predict(data2[linear.feature_names_in_])
linear_final['Item_Outlet_Sales'] = linear_pred
display(linear_final)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Random Forest Regression

# COMMAND ----------

predictors = [x for x in data.columns if x not in [target]+IDcol]
rf = RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4)
modelfit(rf, data, predictors, target, IDcol)

# COMMAND ----------

rf_final = data2[['Item_Identifier','Outlet_Identifier']]
rf_pred = rf.predict(data2[rf.feature_names_in_])
rf_final['Item_Outlet_Sales'] = rf_pred
display(rf_final)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ridge Regression

# COMMAND ----------

predictors = [x for x in data.columns if x not in [target]+IDcol]
ridge = Ridge(alpha=0.05)
modelfit(ridge, data, predictors, target, IDcol)

# COMMAND ----------

ridge_final = data2[['Item_Identifier','Outlet_Identifier']]
ridge_pred = ridge.predict(data2[ridge.feature_names_in_])
ridge_final['Item_Outlet_Sales'] = ridge_pred
display(ridge_final)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Random Forest Regression (Hyperparameter -1)

# COMMAND ----------

predictors = [x for x in data.columns if x not in [target]+IDcol]
rf2 = RandomForestRegressor(n_estimators=400,max_depth=6, min_samples_leaf=100,n_jobs=4)
modelfit(rf2, data, predictors, target, IDcol)

# COMMAND ----------

rf2_final = data2[['Item_Identifier','Outlet_Identifier']]
rf2_pred = rf2.predict(data2[rf2.feature_names_in_])
rf2_final['Item_Outlet_Sales'] = rf2_pred
display(rf2_final)

# COMMAND ----------

# MAGIC %md
# MAGIC ## XGBoost Regression

# COMMAND ----------

# MAGIC %pip install xgboost
# MAGIC
# MAGIC from xgboost import XGBRegressor

# COMMAND ----------

predictors = [x for x in data.columns if x not in [target]+IDcol]
XGB = XGBRegressor()
modelfit(XGB, data, predictors, target, IDcol)


# COMMAND ----------

xgb_final = data2[['Item_Identifier','Outlet_Identifier']]
xgb_pred = XGB.predict(data2[XGB.feature_names_in_])
xgb_final['Item_Outlet_Sales'] = xgb_pred
display(xgb_final)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Random Forest Regression (Hyperparameter-2)

# COMMAND ----------

predictors = [x for x in data.columns if x not in [target]+IDcol]
rf1=RandomForestRegressor(n_estimators=45, max_depth=7, random_state=42)
modelfit(rf1, data, predictors, target, IDcol)

# COMMAND ----------

rf1_final = data2[['Item_Identifier','Outlet_Identifier']]
rf1_pred = rf1.predict(data2[rf1.feature_names_in_])
rf1_final['Item_Outlet_Sales'] = rf1_pred
display(rf1_final)

# COMMAND ----------

# MAGIC %md
# MAGIC ## XGBoost Regression (Hyper Parameter)

# COMMAND ----------

from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300,1000],
    'max_depth': [3, 4, 5,10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0]
}

# Initialize the XGBRegressor
xgb = XGBRegressor()

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(data[predictors], data[target])

# Get the best parameters and best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Fit the best model
modelfit(best_model, data, predictors, target, IDcol)

# COMMAND ----------

best_model_final = data2[['Item_Identifier','Outlet_Identifier']]
best_model_pred = best_model.predict(data2[best_model.feature_names_in_])
best_model_final['Item_Outlet_Sales'] = best_model_pred
display(best_model_final)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gradient Boost Regression

# COMMAND ----------

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0]
}

# Initialize the GradientBoostingRegressor
gbr = GradientBoostingRegressor()

# Initialize GridSearchCV
grid_search_gbr = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)

# Fit GridSearchCV
grid_search_gbr.fit(data[predictors], data[target])

# Get the best parameters and best model
best_params_gbr = grid_search_gbr.best_params_
best_model_gbr = grid_search_gbr.best_estimator_

# Fit the best model
modelfit(best_model_gbr, data, predictors, target, IDcol)

# COMMAND ----------

best_model_gbr_final = data2[['Item_Identifier','Outlet_Identifier']]
best_model_gbr_pred = best_model_gbr.predict(data2[best_model.feature_names_in_])
best_model_gbr_final['Item_Outlet_Sales'] = best_model_gbr_pred
display(best_model_gbr_final)

# COMMAND ----------

# MAGIC %md
# MAGIC ##CATBOOST

# COMMAND ----------

y_train = train['Item_Outlet_Sales']
x_train = train.drop(['Item_Outlet_Sales'],axis=1)

# COMMAND ----------

x_train.head()

# COMMAND ----------

# MAGIC %pip install catboost

# COMMAND ----------

from catboost import CatBoostRegressor, Pool
categorical_features =  np.where(x_train.dtypes == object )[0]

def objective(trial,data=x_train,target=y_train):

    train_x, test_x, train_y, test_y = train_test_split(x_train, y_train, test_size=0.15,random_state=42)
    param = {
        'loss_function': 'RMSE',
        #'task_type': 'GPU',
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-3, 10.0),
        'max_bin': trial.suggest_int('max_bin', 200, 400),
        #'rsm': trial.suggest_uniform('rsm', 0.3, 1.0),
        'subsample': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'learning_rate': trial.suggest_uniform('learning_rate', 0.006, 0.018),
        'n_estimators':  1000,
        'max_depth': trial.suggest_categorical('max_depth', [5,7,9,11,13,15]),
        'random_state': trial.suggest_categorical('random_state', [2020]),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 300),
    }
    model = CatBoostRegressor(**param,cat_features=categorical_features)

    model.fit(train_x,train_y,eval_set=[(test_x,test_y)],early_stopping_rounds=200,verbose=False)

    preds = model.predict(test_x)

    rmse = mean_squared_error(test_y, preds,squared=False)

    return rmse

# COMMAND ----------

Best_trial = {'l2_leaf_reg': 0.001061926310,'max_bin': 322,
 'learning_rate': 0.01081467174,'max_depth': 5,'random_state': 2020,'min_data_in_leaf': 163,
              'loss_function': 'RMSE','n_estimators':  1000}

# COMMAND ----------

from catboost import CatBoostRegressor, Pool
categorical_features =  np.where(x_train.dtypes == object )[0]

model = CatBoostRegressor(**Best_trial,cat_features=categorical_features)
model.fit(x_train, y_train)
test_pred = model.predict(test[x_train.columns])

# COMMAND ----------

test = pd.read_csv('test.csv')

# COMMAND ----------

test

# COMMAND ----------

final = test[['Item_Identifier','Outlet_Identifier']]

# COMMAND ----------

final

# COMMAND ----------

final['Item_Outlet_Sales'] = test_pred
final.reset_index(drop=True,inplace=True)
final
#final.to_csv('catboost.csv')

# COMMAND ----------

final.to_csv('catboost.csv')

# COMMAND ----------

