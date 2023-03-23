# Bike Sharing Demand Prediction -ML regression Project (End to End)

Prepare---> Build ---> Deploy

# Problem Description:

Currently Rental bikes are introduced in many urban cities for the enhancement of mobility comfort. It is important to make the rental bike available and accessible to the public at the right time as it lessens the waiting time. Eventually, providing the city with a stable supply of rental bikes becomes a major concern. The crucial part is the prediction of bike count required at each hour for the stable supply of rental bikes.

# Data Description:

The dataset contains weather information (Temperature, Humidity, Windspeed, Visibility, Dewpoint, Solar radiation, Snowfall, Rainfall), the number of bikes rented per hour and date information.

# Attribute Information:

Date : year-month-day

Rented Bike count - Count of bikes rented at each hour

Hour - Hour of he day

Temperature-Temperature in Celsius

Humidity - %

Windspeed - m/s

Visibility - 10m

Dew point temperature - Celsius

Solar radiation - MJ/m2

Rainfall - mm

Snowfall - cm

Seasons - Winter, Spring, Summer, Autumn

Holiday - Holiday/No holiday

Functional Day - NoFunc(Non Functional Hours),

Fun(Functional hours)

# Important Requirements:

- Python
- Flask
- pickle
- Pipeline

# How to install requirements

`pip install -r requirements.txt`


# Deploy model

To deploy app run this command on prompt

`python app.py`


# Predict using data

- Model accepts data in .json format
- Send request to this `'./predict'` link to predict data

The demo API post request can be post using test.py file

`python test.py`
