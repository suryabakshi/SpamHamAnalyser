import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC

from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

def textPreprocessor1(featureRecord):
    #a.Remove Punctuation
    removePunctuation = [char for char in featureRecord if char not in string.punctuation]
    sentences = ''.join(removePunctuation)
    
    #b.Convert Sentences to Words
    words = sentences.split(" ")
    
    #c. Normalize
    wordNormalized = [word.lower() for word in words]
    
    #d. Remove Stopwords
    finalWords = [word for word in wordNormalized if word not in stopwords.words("english")]
    
    return finalWords

textPreprocessor = pickle.load(open('textPreprocessor.pkl', 'rb'))
model = pickle.load(open('SMS_spamham.pkl', 'rb'))
tfIdfObject = pickle.load(open('tfIdfObject.pkl','rb'))
finalWordVocab= pickle.load(open('word_vocab.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
 

def predict():
    '''
    For rendering results on HTML GUI
    '''
    SMSInput = request.form['SMS']
    
    #1. Preprocess

    preProcessedFeatures = textPreprocessor1(SMSInput)

    #2. BOW transformation
    bowFeature = finalWordVocab.transform(preProcessedFeatures)
   
    #3. TFIDF transformation

    tfIdfFeature = tfIdfObject.transform(bowFeature)
	
    #4. Model Predict

    predLabel = model.predict(tfIdfFeature)[0] 

    return render_template('index.html', prediction_text=' The SMS entered is a {}'.format(predLabel))

if __name__ == "__main__":
    app.run(debug=True)