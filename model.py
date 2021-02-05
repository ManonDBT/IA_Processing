#bibliothèque de base

import requests
import spacy
import fr_core_news_sm
import numpy as np
import pandas as pd
import re
import pickle
import sklearn.externals as extjoblib
import joblib
from joblib import dump

#bibliothèque pour le traitement de texte
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nlp = spacy.load("fr_core_news_sm")

#bibliothèque pour la classification
from scipy import linalg # SVD
from sklearn import decomposition # NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

# ignore ConvergenceWarnings
from  warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

from io import StringIO

#region loading data

df = pd.read_excel('./cr.xlsx')
include = ['text', 'category'] # Only four features
df_ = df[include]

#endregion

#region modification du tableau
# je prends que les colonnes qui m'interresse
col = ['category', 'text']
df = df[col]
df = df[pd.notnull(df['text'])]

df.columns = ['category', 'text']

# je labellise les données category

df['category_id'] = df['category'].factorize()[0]

category_id_df = df[['category', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'category']].values)



#endregion

#region Data Preprocessing

stop_words = stopwords.words('french')


ps = PorterStemmer()


def clean_text(txt):
    txt = txt.lower()  # lowercase
    txt = re.sub("[^a-zA-Z]", " ", txt)  # Remove everything except alphabetical characters
    txt = word_tokenize(txt)  # tokenize (split into list and remove whitespace)

    # initialize list to store clean text
    clean_text = ""

    # iterate over each word
    for w in txt:
        # remove stopwords
        if w not in stop_words:
            # stem=ps.stem(w) #stem
            stem = w
            clean_text += stem + " "
    return clean_text


text_new = []  # declare a list to hold new movies

for cell in df['text']:
    txt = clean_text(cell)
    text_new.append(txt)

# add new info column to the dataframe
df['text'] = text_new

#endregion


#region  preparation modèle

final_stopwords_list = stopwords.words('english') + stopwords.words('french')

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='utf-8', ngram_range=(1, 2), stop_words=final_stopwords_list)
features = tfidf.fit_transform(df.text).toarray()
labels = df.category_id

N = 3
for category, category_id, in sorted(category_to_id.items()):
    features_chi2 = chi2(features, labels == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]



#endregion


#region model

dependent_variable = 'category_id'
x = df[df.columns.difference([dependent_variable])]
y = df[dependent_variable]
lsvc = LinearSVC()


#endregion

# Save your model
joblib.dump(lsvc, 'modelsvc.pkl')
print("Model dumped!")

# Load the model that you just saved
lsvc = joblib.load('modelsvc.pkl')

# Saving the data columns from training
model_columns = list(x.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")

