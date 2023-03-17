# %% [markdown]
# # <b><u> Project Title : Seoul Bike Sharing Demand Prediction </u></b>

# %% [markdown]
# ## <b> Problem Description </b>
# 
# ### Currently Rental bikes are introduced in many urban cities for the enhancement of mobility comfort. It is important to make the rental bike available and accessible to the public at the right time as it lessens the waiting time. Eventually, providing the city with a stable supply of rental bikes becomes a major concern. The crucial part is the prediction of bike count required at each hour for the stable supply of rental bikes.
# 

# %% [markdown]
# ## <b> Data Description </b>
# 
# ### <b> The dataset contains weather information (Temperature, Humidity, Windspeed, Visibility, Dewpoint, Solar radiation, Snowfall, Rainfall), the number of bikes rented per hour and date information.</b>
# 
# 
# ### <b>Attribute Information: </b>
# 
# * ### Date : year-month-day
# * ### Rented Bike count - Count of bikes rented at each hour
# * ### Hour - Hour of he day
# * ### Temperature-Temperature in Celsius
# * ### Humidity - %
# * ### Windspeed - m/s
# * ### Visibility - 10m
# * ### Dew point temperature - Celsius
# * ### Solar radiation - MJ/m2
# * ### Rainfall - mm
# * ### Snowfall - cm
# * ### Seasons - Winter, Spring, Summer, Autumn
# * ### Holiday - Holiday/No holiday
# * ### Functional Day - NoFunc(Non Functional Hours), Fun(Functional hours)

# %% [markdown]
# # Data Dictionary
# 
# **df** = It is dataframe which created from database "SeoulBikeData.csv"
# 
# **dict_rename_col** = It is a dictionary contains name of feature as key and corresponding replacement name as value.
# 
# **all_features** = It is name of all features in the database
# 
# **numerical_features** = It is a list that contains name of all numerical features in the database
# 
# **categorical_features** = It is a list that contains name of all categorical features of database
# 
# **calculate_iqr** = It is function to calculate uppper limit and lower limit of IQR(Inter quartile range)
# 
# **df['day']** = It is a series that contains day number extracted from date feature from df 
# 
# **df['month']** = It is a series that contains month number extracted from date feature from df 
# 
# **df['week_day']** = It is a series that contains week day number extracted from date feature from df
# 
# **convert_hour** = It is a function to convert hour of the day into four time slot
# 
# **remove_features** = It contains name of features to be removed from df
# 
# **add_cat_features** = It contains name of categorical features to be added to list of categorical_features
# 
# **df2** = It is a dataframe which is copy of df
# 
# **df2['time_slot']** = It is a series that contains name of time_slot converted to hours 
# 
# **season_demand** = It is a dataframe which contains name of season and total demand of rental bikes in that season.
# 
# **weekday_demand** = It is a dataframe which contains name of day of the week and total demand of rental bikes in that week day.
# 
# **month_demand** = It is a dataframe which contains name of number of the month and total demand of rental bikes in that month.
# 
# **hour_demand** = It is a dataframe which contains name of number of the hour and total demand of rental bikes in that hour.
# 
# **calc_vif** = It is a function to calculate VIF
# 
# **df1** = It is a dataframe which is copy of dataframe df
# 
# **dependent_var** : It is a list which  contains name of dependent feature
# 
# **independent_var** : It is a list which  contains name of independent feature
# 
# **X** = It is a dataframe which consist all values of features in list of  independent variable
# 
# **y** = It is a dataframe which consist all values of features in list of dependent variable
# 
# 
# 
# 
# 
# 
# 
# 
# 

# %%
# importing essential libraries 

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
#importing data into dataframe
file_path ='SeoulBikeData.csv'
df = pd.read_csv(file_path,encoding = "ISO-8859-1")

# %%
#df.info()

# %%
# checking top 5 rows

#df.head()

# %%
# checking last 5 rows

#df.tail()

# %% [markdown]
# #Checking for duplicated row

# %%
# checking for duplicated rows
#len(df[df.duplicated()])

# %% [markdown]
# #Checking for null/NaN/Missing values and outliers

# %%
#cheking for null values in every column

#df.isnull().sum()

# %%
#Generating descriptive statistics

#df.describe([0.75,0.90,0.95,0.99])

# %%
#rename columns

dict_rename_col = {'Temperature(°C)':'temp',
                   'Humidity(%)':'humidity',
                   'Wind speed (m/s)':'wind_speed',
                   'Visibility (10m)':'visibility',
                   'Dew point temperature(°C)':'dew_point_temperature',
                   'Solar Radiation (MJ/m2)':'solar_radiation',
                   'Rainfall(mm)':'rainfall',
                   'Snowfall (cm)':'snowfall',
                   'Rented Bike Count':'rented_bike_count',
                   'Hour':'hour',
                   'Seasons':'seasons',
                   'Holiday':'holiday',
                   'Functioning Day':'functioning_day',
                   'Date':'date'



                   }

df = df.rename(columns = dict_rename_col)

# %%
#df.head()

# %% [markdown]
# Various features in the list

# %%
#all features
all_features = list(df.columns)



#numerical features

numerical_features = list(df.describe().columns)
numerical_features.pop(1)


#categorical features

categorical_features = [col for col in all_features if col not in numerical_features]


#print(f'Name of all features are as follows : {all_features}')
#print(f'Name of numerical features are as follows : {numerical_features}')
#print(f'Name of categorical features are as follows : {categorical_features}')






# %% [markdown]
# #Checking distribution of dependent feature

# %% [markdown]
#            We can observe that data is postively skewed

# %%
#distrution of dependent variable

#sns.displot(df['rented_bike_count'])

#print(df['rented_bike_count'].skew())

# %% [markdown]
# After giving log transformation,it can be seen that it is now treated

# %%
#sns.displot(np.log(df['rented_bike_count']),bins = 20)

# %% [markdown]
# # Outlier Removal

# %%
def calculate_iqr(col):
  
  '''to calculate up_limit and lower limit of IQR'''
  
  q1 = df[col].quantile(0.25)
  q3 = df[col].quantile(0.75)
  iqr = q3-q1
  up_limit = 1.5 * iqr + q3
  low_limit = q1 - 1.5 * iqr

  return up_limit,low_limit,q1,q3

# %%
# plotting box plot to check outliers

#for col in numerical_features:
  #sns.boxplot(x=df[col])
  #plt.title(f"Title: Box plot for {col}")
  #plt.grid()
  #plt.show()

# %% [markdown]
# Removing outlier from rented_bike_count feature

# %%
up_limit_rbc,low_limit_rbc,q1,q3 = calculate_iqr('rented_bike_count')

up_limit_rbc,low_limit_rbc,q1,q3

# %%
#df['rented_bike_count'].describe()

# %%
df[df['rented_bike_count'] > up_limit_rbc]

# %%
df = df[df['rented_bike_count'] < up_limit_rbc]

# %% [markdown]
# Removing outlier from wind_speed column

# %%
up_limit_ws,low_limit_ws,q1,q3 = calculate_iqr('wind_speed')

#up_limit_ws,low_limit_ws,q1,q3

# %%
# no of outliers in wind_speed feature

#len(df[df['wind_speed']>up_limit_ws])

# %%
df = df[df['wind_speed']<up_limit_ws]

# %%
# plotting box plot after removing outliers

#for col in numerical_features:
  #sns.boxplot(x=df[col])
  #plt.title(f"Title: Box plot for {col}")
  #plt.grid()
  #plt.show()

# %% [markdown]
# Removing outliers from solar radiation

# %%
up_limit_sr,low_limit_sr,q1,q3 = calculate_iqr('solar_radiation')

#up_limit_sr,low_limit_sr,q1,q3

# %%
# no of outliers in solar_radiation feature

#len(df[df['solar_radiation']>up_limit_sr])

# %%
#df['solar_radiation'].describe([0.025,0.75,0.90,0.95,0.975,0.99])

# %%
# trimming of data

df = df[df['solar_radiation']<up_limit_sr]

# %%
#distribution of solar_radiation feature
#sns.displot(df['solar_radiation'])

# %% [markdown]
# Removing outliers from rainfall column

# %%
up_limit_rainfall,low_limit_rainfall,q1,q3 = calculate_iqr('rainfall')

#up_limit_rainfall,low_limit_rainfall,q1,q3

# %%
# no of outliers in rainfall feature

#len(df[df['rainfall']>up_limit_rainfall])

# %%
#df = df[df['rainfall']<up_limit_rainfall]

# %%
df = df[df['rainfall']==up_limit_rainfall]

# %%
#sns.displot(df['rainfall'])

# %% [markdown]
# Removing outliers snowfall

# %%
up_limit_snowfall,low_limit_snowfall,q1,q3 = calculate_iqr('snowfall')

#up_limit_snowfall,low_limit_snowfall,q1,q3

# %%
#df[df['snowfall']>up_limit_snowfall]

# %%
# trimming snowfall feature
df=df[df['snowfall']==up_limit_snowfall]

# %% [markdown]
# # Feature engineering

# %%
#df.info()

# %%

# convert the 'Date' column to datetime format
df['date']= pd.to_datetime(df['date'])

# %%
#extracting day,month,week day and month and creating new column respectively
df['day'] = df['date'].dt.day
df['month'] =df['date'].dt.month
df['week_day'] = df['date'].dt.weekday



# %%
df.drop(columns='date',inplace=True)

# %%
#df.head()

# %%
#hour column has too many categories
# We must convert hour to morning,noon,evening,night

#df['hour'].value_counts()

# %%
def convert_hour(input_hour):

  '''convert hour of the day into four time slot''' 


  if 5 <= input_hour <12:
    return 'morning'
  elif 12<= input_hour < 17:
    return 'afternoon'
  elif 17 <= input_hour < 22:
    return 'evening'
  else:
    return 'night'

# %%
#df['time_slot'] = df['hour'].apply(convert_hour)

# %%
#df['time_slot'].value_counts()

# %%
#checking temperature columns
#it is a continuous feature

#df.temp.value_counts()

# %%
#checking humidity feature

#df.humidity.value_counts()

# %%
#checking wind_speed feature

#df.wind_speed.value_counts()

# %%
#checking visibility feature

#df.visibility.value_counts()

# %%
#checking dew_point_temperature feature

#df.dew_point_temperature.value_counts()

# %%
#df.info()

# %%
#checking solar_radiation feature

#df.solar_radiation.value_counts()

df.drop(columns='solar_radiation',inplace=True)

# %%
#checking rainfall feature

#df.rainfall.value_counts()

#all values are 0, there it will be better to drop this column
df.drop(columns='rainfall',inplace=True)

# %%
#checking snowfall feature

#df.snowfall.value_counts()

#all values are 0, there it will be better to drop this column
df.drop(columns='snowfall',inplace=True)

# %%
#checking seasons feature

#df.seasons.value_counts()

# %%
#checking holiday feature

#df.holiday.value_counts()

# %%
#checking functioning_day feature

#df.functioning_day.value_counts()




# %% [markdown]
# # Univarite analyisis

# %%
remove_features = ['snowfall','rainfall','solar_radiation']

numerical_features = [col for col in numerical_features if col not in remove_features]

# %%
numerical_features

# %%
add_cat_features = ['day','month','week_day']
categorical_features.extend(add_cat_features)

# %%
#categorical_features

# %%
#categorical_features
categorical_features.pop(0)

# %%
#categorical_features.pop(3)

# %%
#categorical_features.pop(2)

# %%
#checking distribution of all numerical features

#for col in numerical_features:
  #sns.displot(df[col])
  #plt.title(f'Title: Distribution plot for {col} with skew {round(df[col].skew(),2)}')


# %%
#categorical_features

# %%
#df['seasons'].value_counts()

# %%
#check distribution of all categorical features

#for col in categorical_features:
  #plt.figure(figsize=(3,2))
  #sns.countplot(x = df[col])
  #plt.show()
  #plt.savefig(f"{col}_countplot.png")
  #plt.close()

# %%
#df.drop(columns=['functioning_day','holiday'],inplace=True)

# %%
#df.info()

# %% [markdown]
# #Bivariate analysis

# %%
#check relation between categorical features and dependent variable

#for col in categorical_features:
  #plt.figure(figsize=(9,6))
  #sns.boxplot(x=df[col],y='rented_bike_count',data=df)
  #plt.show()

# %% [markdown]
# #Multivariate analysis

# %%
#df.info()

# %%
#creating copy of df to do analysis
df2=df.copy()

# %%
#df2.head()

# %%
df2['time_slot'] = df2['hour'].apply(convert_hour)

# %% [markdown]
# No of bikes rented according to time slot

# %% [markdown]
#       Findings : 
#       1. Bikes are mostly rented at night time
#       2. Second most time slot at which bike is rented.
#       3. At afternoon least no of bikes are rented

# %%
#sns.countplot(data = df2,x='time_slot')

# %% [markdown]
# Season and demand of bike

# %% [markdown]
#          It can be seen that demand of bike is in following order:
#          Summer > Autumn > Spring > Winter
#         
# 

# %%
season_demand = pd.DataFrame(data=df2.groupby('seasons')['rented_bike_count'].sum().reset_index())
#season_demand


# %%
#sns.barplot(data = season_demand,x = 'seasons',y='rented_bike_count')

# %% [markdown]
# Relation between temperature and bike demand

# %% [markdown]
# Following are the relationship between temperature and demand of rental bike:
# 1. It can be seen that demand of bike increases with increase in temperature
# 2. When temperature is high demand for bike rental is more during evening.
# 3. When temperature is low demand for bike rental is during morning and night.      
#         
# 

# %%
#sns.scatterplot(data = df2,x='temp',y='rented_bike_count',hue = 'time_slot' )

# %%
#month_temp = pd.DataFrame(data=df2.groupby('month').agg(mean_temperature = ('temp','mean'),mean_bike_demand =('rented_bike_count','mean')).reset_index())
#month_temp

# %%
#plt.figure(figsize=(9,6))
#sns.scatterplot(data=month_temp,y = 'mean_bike_demand',x = 'mean_temperature',hue = 'month')

# %% [markdown]
# Relationship between week day and demand for bike

# %% [markdown]
# It can be seen that almost at any day of the week demand is almost same

# %%
#weekday_demand = pd.DataFrame(data=df2.groupby('week_day')['rented_bike_count'].sum().reset_index())
#weekday_demand


# %%
#sns.barplot(data = weekday_demand,x = 'week_day',y='rented_bike_count')

# %% [markdown]
# Relationship between month and bike demand

# %% [markdown]
#  Key finding related to monthly bike demand:
#  1. Rental bike demand is low on January, February and December.
#  2. Rental bike demand is very high between May to August. 

# %%
month_demand = pd.DataFrame(data=df2.groupby('month')['rented_bike_count'].sum().reset_index())
#month_demand

# %%
#sns.barplot(data = month_demand,x = 'month',y='rented_bike_count')

# %%
#df2.info()

# %% [markdown]
# Relationship between visibilty and bike demand

# %% [markdown]
# There is no clear trend between visibility and bike demand 

# %%
#df2['visibility'].value_counts()

# %%
#plt.figure(figsize=(12,9))
#sns.scatterplot(data = df2 , x='visibility',y ='rented_bike_count')

# %% [markdown]
# Relationship between wind_speed and bike demand

# %% [markdown]
# It can be seen that bike demand usually lowers when wind speed is higher.

# %%
#plt.figure(figsize=(12,9))
#sns.scatterplot(data = df2 , x='wind_speed',y ='rented_bike_count')

# %%
#plt.figure(figsize=(12,9))
#sns.lineplot(data = df2 , x=df['wind_speed'].values,y =df['rented_bike_count'].values)

# %% [markdown]
# Relationship between hour of the day and bike demand

# %%
#plt.figure(figsize=(12,9))
#sns.lineplot(data = df2 , x=df['hour'].values,y =df['rented_bike_count'].values)

# %%
#plt.figure(figsize=(12,9))
#sns.scatterplot(data = df2 , x=df['hour'].values,y =df['rented_bike_count'].values)

# %%
#plt.figure(figsize=(12,9))
#sns.barplot(data = df2 , x=df['hour'].values,y =df['rented_bike_count'].values)

# %%
hour_demand = pd.DataFrame(data=df2.groupby('hour')['rented_bike_count'].sum().reset_index())
#hour_demand

# %% [markdown]
# It can be seen that bike demand rises after 5 AM and peaks at 8 AM, then again rises after 2 PM and peaks at 5PM then demand remain significantly above average demand 6PM and 11PM
# 
# That means in this 11 hours of a day bike demand is most.

# %%
#plt.figure(figsize=(12,9))
#sns.barplot(data = hour_demand , x='hour',y ='rented_bike_count')

# %% [markdown]
# # Checking correlation

# %%
#numerical_features

# %%
#df.corr()

# %%
#heatmap to visualise correlation
#plt.figure(figsize=(9,6))
#sns.heatmap(df.corr(),annot = True)

# %%
#removing dew point because of 96% correlation
df.drop(columns='dew_point_temperature',inplace=True)

# %%
#removing hour because of 96% correlation
#df.drop(columns='hour',inplace=True)

# %% [markdown]
# #VIF analysis

# %%
#df.info()

# %%
all_features=list(df.columns)
#all_features

# %%
#categorical_features.pop(0)

# %%
#numerical_features

# %%
#numerical_features
numerical_features.pop(5)

# %%
# Import library for VIF
#from statsmodels.stats.outliers_influence import variance_inflation_factor

#def calc_vif(X):

    # Calculating VIF
    #vif = pd.DataFrame()
    #vif["variables"] = X.columns
    #vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    #return(vif)

# %%
# check multicollinearity, VIF must be less than 10

#calc_vif(df[[i for i in df.describe().columns if i not in ['rented_bike_count']]])

# %%
#df.info()

# %%
#print(len(numerical_features))
#print(len(categorical_features))

# %%
#numerical_features

# %%
#categorical_features

# %% [markdown]
# #Categorical encoding

# %%
df1=df.copy()

# %%
#df1.info()

# %%
#df1.head()

# %%
#df1.tail()

# %%
df1=df1.reset_index()

# %%
df1.drop(columns='index',inplace=True)

# %%
#df1.head()

# %%
#df1.tail()

# %%
df1 = pd.get_dummies(data = df1,columns =categorical_features)

# %%
#df1.info()

# %% [markdown]
# #Building ML model

# %%
#df1.columns.to_list()

# %%
dependent_var = 'rented_bike_count'
independent_var = list(set(df1.columns.to_list()) -{dependent_var})

# %%
#print(len(independent_var))

# %%
X = df1[independent_var]
y =np.log(df1[dependent_var]+1)

# %%
#sns.displot(y)

# %% [markdown]
# After yeo johnson transformation accuracy reduced to 83.5%, so reverting back to log transformation

# %%
# apply yeo johnson accuracy reduced 
#from scipy import stats

#X = df1[independent_var]
#m , lamda= stats.yeojohnson(df1[dependent_var])
#y = m

# %%
#y

# %% [markdown]
# #Data Dictionay (continued)
# 
# 
# **X_train, X_test, y_train, y_test** : Split of dataframe into test and train set, dataframe after 20%
# 
# **convert_log_toactual** : It is a function that converts log transformed value of "y" into actual value
# 
# **reg** = It is Linear regression model
# 
# **r2_score_lr** : It contains r2 score of reg (Linear regression) model.
# 
# **aj_r2_score_lr** : It contains adjusted r2 score of reg (Linear regression) model.
# 
# **lasso** = It is Lasso linear regression model
# 
# **r2_score_lasso**: It contains r2 score of Lasso (Linear regression) model.
# 
# **adj_r2_score_lasso** : It contains adjusted r2 score of reg (Linear regression) model.
# 
# **lasso_regressor** : It is It is lasso linear regression model with hyperparameter tuning with CV
# 
# **lasso_optimum_model** : It is lasso ML model with best estimator lasso linear regression model with hyperparameter tuning with CV
# 
# **r2_score_lasso_cv** : It contains r2 score of Lasso (Linear regression) model with cv
# 
# **aj_r2_score_lasso_cv** : It contains ajusted r2 score of Lasso (Linear regression) model with cv
# 
# **rf** : It is a random forest regessor ML model.
# 
# **rf_cv** : It is a random forest regessor ML model with hyper parameter tuning with CV
# 
# **rf_cv_optimal_model** : It is random forest ML model with best estimator of random forest regression model with hyperparameter tuning with CV
# 
# **r2_rf_cv** : It contains r2 score of random forest regessor ML model with hyper parameter tuning with CV
# 
# **xgb** : It is a XG Boost regressor ML model.
# 
# **r2_xgb** : It is r2 score of xgb model
# 
# **adj_r2_xgb** : It is adjusted r2 score of xgb model
# 
# **xgb_cv**: It is xgb ML model with hyper parameter tuning with CV  
# 
# **xgb_cv_optimal**: It is xgb_cv model with best estimator
# 
# **r2_xgb_cv** : It is r2 score of xgb_cv model.
# 
# **aj_r2_xgb_cv** : It is adjusted r2 score of xgb_cv_model
# 
# 
# 
# 
# 
# 

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# %%
#use this columns to arrange pattern after dummyfying
pr_col = X_train.columns

# %%
#X_train.info()

# %%
#X_train.head()

# %%
#X_test.head()

# %%
#y_train.head()

# %%
#y_test.head()

# %%
#print(X_train.shape,y_train.shape)
#print(X_test.shape,y_test.shape)

# %%
#sns.displot(y_train)

# %% [markdown]
# Revert np log value

# %%
#np.log(5)+1

# %%
#np.exp(2.6094379124341005-1)

# %%
def convert_log_toactual(input_pred):
  '''reverse log transformed value to actual value'''
  y_actual = np.exp(input_pred-1)
  return y_actual  

# %%
from sklearn.preprocessing import MinMaxScaler
#Transforming data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


import xgboost as xg

# %%
xgb= xg.XGBRegressor(objective ='reg:squarederror',
                  n_estimators = 800)

xgb.fit(X_train, y_train)

# %%
xgb.score(X_train, y_train)

# %%
xgb.score(X_test, y_test)

# %%
y_pred_xgb = xgb.predict(X_test)

# %%
#y_pred_xgb

# %%
#actual predicted value
y_pred_xgb_actual = convert_log_toactual(y_pred_xgb)

# %%
from sklearn.metrics import mean_squared_error

MSE  = mean_squared_error((y_test), (y_pred_xgb))

#print("MSE :" , MSE)

RMSE = np.sqrt(MSE)
#print("RMSE :" ,RMSE)

# %%
from sklearn.metrics import r2_score
r2_xgb = r2_score((y_test),(y_pred_xgb))
#print("R2 :" ,r2_xgb)
#print("Adjusted R2 : ",1-(1-r2_score((y_test), (y_pred_xgb)))*((X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1)))

# %%
#r2_xgb
aj_r2_xgb = 1-(1-r2_score((y_test), (y_pred_xgb)))*((X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1))
#aj_r2_xgb

# %% [markdown]
# XG boost regression model hyper parameter tuning

# %%
#Cross validation using grid search cv

from sklearn.model_selection import GridSearchCV

parameters_dict = {
    'n_estimators' : [600,700,800,900,1000,1200,1400]
    }
xgb= xg.XGBRegressor(objective ='reg:squarederror')
xgb_cv = GridSearchCV(xgb,param_grid = parameters_dict,cv = 5,verbose = 2)
xgb_cv.fit(X_train,y_train)

# %%
#best params for XG boost
xgb_cv_optimal = xgb_cv.best_estimator_
#xgb_cv_optimal

# %%
#Score for train set

xgb_cv_optimal.score(X_train, y_train)

# %%
#Score for test set

xgb_cv_optimal.score(X_test, y_test)

# %%
y_pred_xgb_cv = xgb_cv_optimal.predict(X_test)

# %%
y_pred_xgb_cv_actual = convert_log_toactual(y_pred_xgb_cv)

# %%
from sklearn.metrics import mean_squared_error

MSE  = mean_squared_error((y_test), (y_pred_xgb_cv))

#print("MSE :" , MSE)

RMSE = np.sqrt(MSE)
#print("RMSE :" ,RMSE)

# %%
from sklearn.metrics import r2_score
r2_xgb_cv = r2_score((y_test),(y_pred_xgb_cv))
#print("R2 :" ,r2_xgb)
#print("Adjusted R2 : ",1-(1-r2_score((y_test), (y_pred_xgb_cv)))*((X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1)))

# %%
#r2_xgb_cv
aj_r2_xgb_cv = 1-(1-r2_score((y_test), (y_pred_xgb_cv)))*((X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1))
#aj_r2_xgb_cv

# %% [markdown]
# # Creating data frame name of ML model, r2 score, adjusted r2 score

# %%
#adj_r2_score_lasso

# %%
#score_df = pd.DataFrame()
#score_df['name_of_model'] = ['linear regression','lasso regression','lasso regression cv','random forest','random forest cv','xg boost','xg boost cv']
#score_df['r2_score'] = [r2_score_lr,r2_score_lasso,r2_score_lasso_cv,r2_rf,r2_rf_cv,r2_xgb,r2_xgb_cv]
#score_df['adj_r2_score'] = [aj_r2_score_lr,adj_r2_score_lasso,aj_r2_score_lasso_cv,aj_r2_rf,aj_r2_rf_cv,aj_r2_xgb,aj_r2_xgb_cv]

# %%
#score_df

# %% [markdown]
# **Conclusion** : It is found out that demand for bike rises with rise in temperature. At night demand for rental bike is most, In summer season the demand for rental bike is most, In monthly period it is seen that rental bike demand is low on January, February and December and high between may to august, It can be seen that bike demand rises after 5 AM and peaks at 8 AM, then again rises after 2 PM and peaks at 5PM then demand remain significantly above average demand 6PM and 11PM.That means in this 11 hours of a day bike demand is most. *XG boost regression model* can predict rental bike demand with *94.14% accuracy*.

# %%


# %% [markdown]
# # Deploying model

# %%
import pickle
# Pickle the trained model and save it to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(xgb_cv_optimal, file)

# %%

# Load the pickled model
with open('model.pkl', 'rb') as f:
    mdl = pickle.load(f)

# Make a prediction on new data
new_data = X_train[0].reshape(1,-1)
prediction = mdl.predict(new_data)

#predicted_y = np.exp(prediction) - 1

#print(prediction)


# %% [markdown]
# # convert input to model input format

# %%
def inp_mdl_inp(user_input):
    dum_list = pr_col

    # Convert categorical features to numerical using pd.get_dummies()
    user_input_df = pd.DataFrame(user_input, index=[0])
    user_input_df = pd.get_dummies(user_input_df,columns =categorical_features)

    # Reorder the columns to match the order in the training data
    user_input_df = user_input_df.reindex(columns=dum_list, fill_value=0)

    # Make a prediction on new data
    new_data = scaler.transform(user_input_df)
    prediction = mdl.predict(new_data.reshape(1,-1))
    
    #Convert the predicted values back to the original format
    predicted_y = np.exp(prediction) - 1
    
    return predicted_y[0]

# %%
x ={
 'hour': 14,
 'temp': 3.0,
 'humidity': 26,
 'wind_speed': 2.0,
 'visibility': 2000,
 'seasons': 'Winter',
 'holiday': 'No Holiday',
 'functioning_day':'Yes',
 'day': 12,
 'month': 1,
 'week_day': 3}

# %%
inp_mdl_inp(x)


