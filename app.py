from flask import Flask, render_template, request
import jsonify
import requests
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
nltk.download(['punkt', 'wordnet'])

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


app = Flask(__name__)
model = pickle.load(open('cyber_bullying_prediction_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
tfidrizer = pickle.load(open('tfidrizer.pkl', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


'''def tokenize(text):

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens'''


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        msg = request.form['msg']

        tfidf = TfidfTransformer()

        X_test_counts = vectorizer.transform([msg])
        X_test_tfidf = tfidrizer.transform(X_test_counts)

        prediction = model.predict(X_test_tfidf)
        output=prediction[0]
        print(output)
        if output == '1':
            return render_template('index.html',prediction_text="It contains wrong words")
        else:
            return render_template('index.html',prediction_text="It is all right")
    else:
        return render_template('index.html')

if __name__=="__main__":
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    app.run(debug=True)