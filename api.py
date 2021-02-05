# Dependencies
from flask import Flask, request, jsonify
import requests
import sklearn.externals as extjoblib
import joblib
import traceback
import pandas as pd
import numpy as np


# Your API definition
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if modelsvc:
        try:
            url = "http://0.0.0.0:1234/"

            data = requests.get(url + 'datas').json()
            data2 = data['Data']
            df = pd.DataFrame(data2)
            df = df['compte_rendu']
            print(df)
            model =modelsvc.fit(df)
            prediction = list(modelsvc.predict(model))

            return jsonify({'prediction': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')

if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    modelsvc = joblib.load("modelsvc.pkl") # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
    print ('Model columns loaded')

    app.run(port=port, debug=True)