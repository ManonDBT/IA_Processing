# Import dependencies
from flask import Flask, request, Response, jsonify
import requests
import pandas as pd
import numpy as np
from joblib import dump, load



#region loading data
url = "http://0.0.0.0:1234/"

data= requests.get(url+'datas').json()
data2 = data['Data']
df = pd.DataFrame(data2)
print(df)



#region Save your model

dump(model, 'model.joblib')
print("Model dumped!")

#endregion

#region Chargement du modèle
model = load('model.joblib')

#endregion

#region Sauvegarde des données de la colonne depuis l'entrainement
model_columns = list(x.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")
#endregion