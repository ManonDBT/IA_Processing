# Import dependencies
from flask import Flask, request, Response, jsonify
import requests
import pandas as pd
import numpy as np

#region loading data

url = "http://0.0.0.0:1234/"

data= requests.get(url+'datas').json()
data2 = data['Data']
df = pd.DataFrame(data2)
print(df)

#endregion

#region Data Preprocessing


#endregion


#region Logistic Regression classifier/ MODEL


#endregion


#region Save your model
from sklearn.externals import joblib
joblib.dump(lr, 'model.pkl')
print("Model dumped!")

#endregion

#region Load the model that you just saved
lr = joblib.load('model.pkl')

#endregion

#region Saving the data columns from training
model_columns = list(x.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")
#endregion