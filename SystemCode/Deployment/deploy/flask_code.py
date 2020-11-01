# Serve model as a flask application
import pickle
import numpy as np
from flask import Flask, request

scaler_load = None
model_load = None
app = Flask(__name__)

def load_scaler():
    global scaler_load
    # scaler variable refers to the global variable
    with open('./diabetes-scaler.pkl', 'rb') as scaler_pkl:
        scaler_load = pickle.load(scaler_pkl)

def load_model():
    global model_load
    # model variable refers to the global variable
    with open('./diabetes-knn-model.pkl', 'rb') as model_pkl:
        model_load = pickle.load(model_pkl)


@app.route('/')
def home_endpoint():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def get_prediction():
    # Works only for a single sample
    if request.method == 'POST':
        data = request.get_json()  # Get data posted as a json
        data = np.array(data)[np.newaxis, :]  # converts shape from (4,) to (1, 4)
        data = scaler_load.transform(data)
        prediction = model_load.predict(data)  # runs globally loaded model on the data
    return str(prediction[0])


if __name__ == '__main__':
    load_scaler() # load scaler at the beginning once only
    load_model()  # load model at the beginning once only
    app.run(host='0.0.0.0', port=80)