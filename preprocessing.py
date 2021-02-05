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



