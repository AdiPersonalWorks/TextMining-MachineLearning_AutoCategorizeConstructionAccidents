'''

Code to predict the causes using a pipeline algorithm
@author : Team

'''

# Import all packages
import random
import sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn import metrics
import scipy
import os
import json
import unicodedata
import string
import io
import textmining
import pickle

import unicodedata
from nltk import *
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

# Import the file which we will use for classification
oshadocs=pd.read_csv("c:/OSHA_Preprocessed.csv")

titles = oshadocs.Title
cause = oshadocs.Cause

oshadocs=pd.read_csv("../Datasets/OSHA_Preprocessed.csv")

labelled=[]
for row in oshadocs.iterrows():
    index, data = row
    labelled.append(data.tolist())

X_train = titles[0:3558]
y_train = cause[0:3558]
X_test = titles[3559:len(titles)]

# PIPELINE
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier

classifier = Pipeline([('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))]) # Just to keep it scalable.
classifier.fit(X_train, y_train)
predicted = classifier.predict(X_test)

# Saving the classifier (PICKLING)
save_classifier = open("pipeline.pickle","wb")
pickle.dump(classifier,save_classifier)
save_classifier.close()

cause[3559:len(cause)] = predicted

import csv
with open('../Datasets/OSHA_Predicted_Causes.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(zip(cause, titles))