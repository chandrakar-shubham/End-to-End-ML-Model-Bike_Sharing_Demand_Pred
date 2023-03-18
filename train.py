
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

def import_file(file_path):
  df = pd.read_csv(file_path,encoding = "ISO-8859-1")
  return df


#Importing files
file_path = 'SeoulBikeData.csv'
df = import_file(file_path)

def rename_col(df,dict_col_rename):
  df = df.rename(columns = dict_col_rename)
  return df
  

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
                   'Date':'date'}

# rename columns
df = rename_col(df,dict_rename_col)



#segregate features

def seg_features(df):
  all_features = list(df.columns)

  numerical_features = list(df.describe().columns)
  numerical_features.pop(1)
  categorical_features = [col for col in all_features if col not in numerical_features]
  return numerical_features,categorical_features

numerical_features,categorical_features = seg_features(df)


def calculate_iqr(col):
  
  '''to calculate up_limit and lower limit of IQR'''
  
  q1 = df[col].quantile(0.25)
  q3 = df[col].quantile(0.75)
  iqr = q3-q1
  up_limit = 1.5 * iqr + q3
  low_limit = q1 - 1.5 * iqr

  return up_limit,low_limit,q1,q3

up_limit_rbc,low_limit_rbc,q1,q3 = calculate_iqr('rented_bike_count')


df[df['rented_bike_count'] > up_limit_rbc]

df = df[df['rented_bike_count'] < up_limit_rbc]

up_limit_ws,low_limit_ws,q1,q3 = calculate_iqr('wind_speed')

df = df[df['wind_speed']<up_limit_ws]

up_limit_sr,low_limit_sr,q1,q3 = calculate_iqr('solar_radiation')

df = df[df['solar_radiation']<up_limit_sr]

up_limit_rainfall,low_limit_rainfall,q1,q3 = calculate_iqr('rainfall')

df = df[df['rainfall']==up_limit_rainfall]

up_limit_snowfall,low_limit_snowfall,q1,q3 = calculate_iqr('snowfall')


df=df[df['snowfall']==up_limit_snowfall]
# convert the 'Date' column to datetime format
df['date']= pd.to_datetime(df['date'])
df['day'] = df['date'].dt.day
df['month'] =df['date'].dt.month
df['week_day'] = df['date'].dt.weekday



df.drop(columns='date',inplace=True)


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

df.drop(columns='solar_radiation',inplace=True)

#all values are 0, there it will be better to drop this column
df.drop(columns='rainfall',inplace=True)

#all values are 0, there it will be better to drop this column
df.drop(columns='snowfall',inplace=True)


remove_features = ['snowfall','rainfall','solar_radiation']

numerical_features = [col for col in numerical_features if col not in remove_features]

numerical_features

add_cat_features = ['day','month','week_day']
categorical_features.extend(add_cat_features)

categorical_features.pop(0)

df2=df.copy()

# %%
df2['time_slot'] = df2['hour'].apply(convert_hour)

season_demand = pd.DataFrame(data=df2.groupby('seasons')['rented_bike_count'].sum().reset_index())

month_demand = pd.DataFrame(data=df2.groupby('month')['rented_bike_count'].sum().reset_index())

# %%
hour_demand = pd.DataFrame(data=df2.groupby('hour')['rented_bike_count'].sum().reset_index())


df.drop(columns='dew_point_temperature',inplace=True)


all_features=list(df.columns)

numerical_features.pop(5)

df1=df.copy()

df1=df1.reset_index()

df1.drop(columns='index',inplace=True)

df1 = pd.get_dummies(data = df1,columns =categorical_features)

dependent_var = 'rented_bike_count'
independent_var = list(set(df1.columns.to_list()) -{dependent_var})

X = df1[independent_var]
y =np.log(df1[dependent_var]+1)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)


pr_col = X_train.columns

with open('pr_col.txt', 'w') as file:
    for item in pr_col:
        file.write("%s\n" % item)

with open('categorical_features.txt', 'w') as file:
    for item in categorical_features:
        file.write("%s\n" % item)

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

# Dumping scaler object using pickle
import pickle
with open('scaler.pkl', 'wb') as f:
   pickle.dump(scaler, f)


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

RMSE = np.sqrt(MSE)

from sklearn.metrics import r2_score
r2_xgb = r2_score((y_test),(y_pred_xgb))


aj_r2_xgb = 1-(1-r2_score((y_test), (y_pred_xgb)))*((X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1))


from sklearn.model_selection import GridSearchCV

parameters_dict = {
    'n_estimators' : [600,700,800,900,1000,1200,1400]
    }
xgb= xg.XGBRegressor(objective ='reg:squarederror')
xgb_cv = GridSearchCV(xgb,param_grid = parameters_dict,cv = 5,verbose = 2)
xgb_cv.fit(X_train,y_train)

xgb_cv_optimal = xgb_cv.best_estimator_

xgb_cv_optimal.score(X_train, y_train)

xgb_cv_optimal.score(X_test, y_test)

y_pred_xgb_cv = xgb_cv_optimal.predict(X_test)

y_pred_xgb_cv_actual = convert_log_toactual(y_pred_xgb_cv)

from sklearn.metrics import mean_squared_error

MSE  = mean_squared_error((y_test), (y_pred_xgb_cv))

RMSE = np.sqrt(MSE)

from sklearn.metrics import r2_score
r2_xgb_cv = r2_score((y_test),(y_pred_xgb_cv))

aj_r2_xgb_cv = 1-(1-r2_score((y_test), (y_pred_xgb_cv)))*((X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1))


# Pickle the trained model and save it to a file
with open('model.pkl', 'wb') as file:
  
  pickle.dump(xgb_cv_optimal, file)
