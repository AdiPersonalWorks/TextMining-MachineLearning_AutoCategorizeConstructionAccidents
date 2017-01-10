'''
Initial code to predict causes and test SVM and MultiNomialNB for accuracy
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
masia=pd.read_csv("../Datasets/OSHA_Preprocessed.csv")

titles = masia.Title
cause = masia.Cause

# Creating a Term Document Matrix
tdm = textmining.TermDocumentMatrix()

for text in titles:
    tdm.add_doc(text)

tdm.write_csv("OSHA_TDM.csv")
print "OSHA Term Document Matrix Created" 

# Creating an easier variable like tfidf and assigning it the function
tfidf = TfidfVectorizer()

# Creating the TF-IDF
tfs = tfidf.fit_transform(titles)

# View sample TF-IDF values
tfs.data

# Viewing the tfs object
# The words are assigned numerical values. To get the actual words, 
# use the get_feature_names function
feature_names = tfidf.get_feature_names()

### Print out the Feature Names, the TF-IDF score and the Word Index
##for col in tfs.nonzero()[1]:
##    print feature_names[col], ' - ', tfs[0, col], ' - ', tfs.indices[col]
    
# Writing tf-idf into a file. 
# It's a sparse matrix of the type scipy.sparse.csr.csr_matrix, hence, has to be carefully handled
    
scipy.io.mmwrite("tf_idf.mtx.txt", tfs, comment='', field=None, precision=None)
print("TFIDF Created")

######################################################
masia=pd.read_csv("../Datasets/OSHA_Preprocessed.csv")

labelled=[]
for row in masia.iterrows():
    index, data = row
    labelled.append(data.tolist())

# Create a function which will store the TF-IDF for each of the abstracts which we will then use as features to train upon
def create_tfidf_training_data(docs):
    
    """
    Creates a document corpus list (by stripping out the
    class labels), then applies the TF-IDF transform to this
    list. 

    The function returns both the class label vector (y) and 
    the corpus token/feature matrix (X).
    """
    
    # Create the training data class labels
    y = [d[0] for d in docs]

    # Create the document corpus list
    corpus = [d[1] for d in docs]

    # Create the TF-IDF vectoriser and transform the corpus
    vectorizer = TfidfVectorizer(min_df=1)
    X = vectorizer.fit_transform(corpus)
    return X, y
    

# Separate out the X (predictors) and the y (response)    
X, y = create_tfidf_training_data(labelled)

# Create training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Number of features in the Training Set 
len(X_train.data)

# Number of features in the Test Set
len(X_test.data)

# Number of documents in Training set
len(y_train)

# Number of documents in Training set
len(y_test)


################# SUPPORT VECTOR MACHINE###
def train_svm(X, y):
    """
    Create and train the Support Vector Machine.
    """
    svm = SVC(C=1000000.0, gamma=0.47, kernel='rbf')
    svm.fit(X, y)
    return svm

# Train the SVM - using all the data
svm = train_svm(X_train, y_train)

# Predict on the Test set
pred = svm.predict(X_test)

# Print the classification rate
print(svm.score(X_test, y_test))

labels = list(set(y_train))

# Print the confusion matrix
import matplotlib.pyplot as plt
import pylab as pl
cm = confusion_matrix(y_test, pred, labels)
print(cm)

########## MULTINOMIAL NAIVE BAYES#################
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train, y_train)

# Predict on test data
pred = clf.predict(X_test)

# Print the classification rate
print(clf.score(X_test, y_test))

# Print the confusion matrix
import matplotlib.pyplot as plt
import pylab as pl
cm = confusion_matrix(y_test, pred, labels)
print(cm)   