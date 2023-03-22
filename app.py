from flask import Flask, jsonify, request
import pickle
import pandas as pd
import numpy as np
from train import *
import json
app = Flask(__name__)

# load the saved model
with open('final_pipe.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def homepage():
    return 'hello world'

# define API endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # get data from request
    data = request.get_json()

    data1 = json.loads(data)

    # convert data to pandas dataframe
    df = pd.DataFrame(data1)

    # make predictions
    predictions = model.predict(df)

    # convert log-transformed predictions to actual values
    predictions = np.exp(predictions)-1

    # return predictions as JSON
    return jsonify({'predictions': predictions.tolist()})

print("app.py executed")

if __name__ == '__main__':
    app.run()

print("app.py executed successfully")