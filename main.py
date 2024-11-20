from flask import Flask, redirect, url_for, render_template, jsonify, request
import numpy as np
import pickle
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

def stemming(content):
  port_stem = PorterStemmer()
  stemmed = re.sub('[^a-zA-Z]', ' ', content) # substitute all non alphabets to blank
  stemmed = stemmed.lower() # convert all letters to lower case
  stemmed = stemmed.split() # convert to list
  stemmed = [port_stem.stem(word) for word in stemmed if not word in stopwords.words('english')]
  stemmed = ' '.join(stemmed)
  return stemmed

def preprocess_input(data):
  data['content'] = data['content'].apply(stemming)
  x = data['content'].values # returns numpy array of content column

  vectorizer = pickle.load(open('vectorizer.pk', 'rb'))
  x = vectorizer.transform(x)
  return x

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/next", methods=["POST"])
def classify():
    # Load the model and tokenizer
    model = pickle.load(open('model.sav', 'rb'))
    text = request.json['data']
    print(text)

    text = np.array([text])
    text = pd.DataFrame(text, columns=['content'])
    text = preprocess_input(text)

    prediction = model.predict(text)
    prediction = int(prediction[0])
    return jsonify({'label' : prediction})

if __name__ == "__main__":
    app.run()