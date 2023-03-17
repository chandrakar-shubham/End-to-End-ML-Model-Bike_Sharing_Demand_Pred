from flask import Flask, request
import pickle

from train import pr_col, categorical_features, scaler

app = Flask(__name__)

# Load the pickled model when the application starts
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

def inp_mdl_inp(user_input):
    dum_list = pr_col

    # Convert categorical features to numerical using pd.get_dummies()
    user_input_df = pd.DataFrame(user_input, index=[0])
    user_input_df = pd.get_dummies(user_input_df,columns =categorical_features)

    # Reorder the columns to match the order in the training data
    user_input_df = user_input_df.reindex(columns=dum_list, fill_value=0)

    # Make a prediction on new data
    new_data = scaler.transform(user_input_df)
    prediction = model.predict(new_data.reshape(1,-1))
    
    #Convert the predicted values back to the original format
    predicted_y = np.exp(prediction) - 1
    
    return predicted_y[0]

@app.route('/', methods=['GET'])
def home():
    return 'Bike Sharing Bike Prediction Regression Model Deployment'

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from user
    input_data = request.json

    # Use the pickled model to make predictions
    prediction = model.predict(input_data)

    # Return the predicted value as JSON
    return {'prediction': prediction.tolist()}

if __name__ == '__main__':
    app.run(debug=True)
