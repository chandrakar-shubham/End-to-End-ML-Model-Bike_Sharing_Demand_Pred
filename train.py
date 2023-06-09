from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xg
import pandas as pd
import numpy as np
import pickle


def convert_log_toactual(input_pred):
  '''reverse log transformed value to actual value'''
  y_actual = np.exp(input_pred-1)
  return y_actual 

def transform_date(X):
    X_ = X.copy()
    X_['date'] = pd.to_datetime(X_['date'],format="%d/%m/%Y")  #, format='"%d/%m/%Y"'
    X_['day'] = X_['date'].dt.day
    X_['week_day'] = X_['date'].dt.weekday
    X_['month'] = X_['date'].dt.month
    X_ = X_.drop(columns='date')
    return X_


# Define a function to remove outliers from a numerical column using the interquartile range (IQR) method
def remove_outliers(df,column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    df = df[df[column] < (Q3 + 1.5*IQR)]
    return df
    


#parameter for xgboost cv
parameters_dict = {
    'xgb__n_estimators' : [600,700,800,900,1000,1200,1400]
    }

xgb= xg.XGBRegressor(objective ='reg:squarederror',n_estimators=1400)

file_path = '../data/SeoulBikeData.csv'
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

columstokeep=["hour","temp","humidity","wind_speed","visibility","seasons","holiday","functioning_day","date"]

num_col = ["temp","humidity","wind_speed","visibility"]

cat_col = ["hour","seasons","holiday","functioning_day","day","month",'week_day']
# Load the data into a pandas dataframe

import os
import pandas as pd

# Get the absolute path of the script file
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the file path for the data file
data_path = os.path.join(script_dir, 'data', 'SeoulBikeData.csv')

# Load the data from the file
df = pd.read_csv(data_path, encoding='ISO-8859-1')


rel_cols = ['Date', 'Rented Bike Count', 'Hour', 'Temperature(°C)', 'Humidity(%)',
       'Wind speed (m/s)', 'Visibility (10m)', 'Seasons',
       'Holiday', 'Functioning Day']

df = df[rel_cols]

# rename columns
df = df.rename(columns= dict_rename_col)

df = df.pipe(remove_outliers,'rented_bike_count').pipe(remove_outliers,'temp').pipe(remove_outliers,'humidity').pipe(remove_outliers,'wind_speed').pipe(remove_outliers,'visibility')

df = df.reset_index().drop(columns='index')

X = df.drop(columns='rented_bike_count')
y = np.log(df['rented_bike_count']+1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

# define preprocessing steps for categorical and numerical columns
cat_preprocessor = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

num_preprocessor = Pipeline([
    ('scaler', StandardScaler())
])

# create column transformer to apply preprocessing steps to appropriate columns
preprocessor = ColumnTransformer([
    ('cat', cat_preprocessor, cat_col),
    ('num', num_preprocessor, num_col)
])

processor = Pipeline([
        ('date_features', FunctionTransformer(transform_date)),
        ('preprocessor', preprocessor)
        ])

print("train.py executed")
