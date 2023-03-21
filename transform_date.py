import pandas as pd
def transform_date(X):
    X_ = X.copy()
    X_['date'] = pd.to_datetime(X_['date'],format="%d/%m/%Y")  #, format='"%d/%m/%Y"'
    X_['day'] = X_['date'].dt.day
    X_['week_day'] = X_['date'].dt.weekday
    X_['month'] = X_['date'].dt.month
    X_ = X_.drop(columns='date')
    return X_