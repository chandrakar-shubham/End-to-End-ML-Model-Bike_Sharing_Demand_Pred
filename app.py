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

# define API endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # get data from request
    data = request.get_json(force=True)

    data1 = json.loads(data)

    # convert data to pandas dataframe
    df = pd.DataFrame.from_dict(data1,index=[0])

    # make predictions
    predictions = model.predict(df)

    # convert log-transformed predictions to actual values
    predictions = np.exp(predictions)-1

    # return predictions as JSON
    return jsonify({'predictions': predictions.tolist()})

print("app.py executed")
