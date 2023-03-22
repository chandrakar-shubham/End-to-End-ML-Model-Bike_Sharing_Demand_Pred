import requests
import json

url = "http://127.0.0.1:5000/predict"
headers = {"Content-Type": "application/json"}

data = '{"date": ["20/01/2018","21/01/2019"],"hour": [3,4],"temp": [-0.9,10],"humidity": [65,43],"wind_speed": [0.2,0.4],"visibility": [1359,1600],"seasons": ["Winter","Winter"],"holiday": ["No Holiday","No Holiday"],"functioning_day": ["Yes","Yes"]}'

response = requests.post(url, headers=headers,json = data) #data=json.dumps(data)

print(response.status_code)  # Print the HTTP status code returned by the server
print(response.json())  # Print the response data returned by the server