'''

Code to pre-process the original data
@author: TEAM

'''

##### ALL IMPORTS #####
import nltk
import os
import json
import unicodedata
import string
import io
import pandas as pd
import textmining
import sklearn
import numpy as np
global collections
import collections
global operator
import operator
global create_tag_image
global make_tags
global LAYOUTS
global get_tag_counts

from nltk import *
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import stem

##from sklearn.feature_extraction.text import CountVectorizer
from pytagcloud import create_tag_image, make_tags, LAYOUTS
from pytagcloud.lang.counter import get_tag_counts

print('Finished importing all libraries')

# Read data
data=pd.read_csv("../Datasets/OSHA_CauseTitle.csv")
titles = data.Title
##titles = titles[1:10]
cause = data.Cause

# Remove all the punctuations from the text
text_nopunc = []
exclude = set(string.punctuation)
for title in titles:
    npunc ="".join(ch for ch in title if ch not in exclude)
    text_nopunc.append(npunc)
print("Punctuations Removed")

# Convert to lower case
text_lower = []
for title in text_nopunc:
    lowtext=title.lower()
    text_lower.append(lowtext)
print("Lower Case Text")

# Create a stopword list from the standard list of stopwords available in nltk
stop_words = stopwords.words('english')
##print(stop)

# Remove all these stopwords from the text
nstop = []
for title in text_lower:
    words = word_tokenize(title)
    filtered_sentence = [w for w in words if not w in stop_words]
    filtered_sentence = "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in filtered_sentence]).strip()
    nstop.append(filtered_sentence)
print("Stop Words Removed")

# Lemmatizing - WordNetLemmatizer
wnl = nltk.WordNetLemmatizer()
nstem = []
for title in nstop:
    stemmed = []
    words = word_tokenize(title)
    w_stem = " ".join([wnl.lemmatize(w) for w in words])
    nstem.append(w_stem)
print("Lemmatizing Done")
    
import csv
with open('../Datasets/OSHA_Preprocessed.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(zip(cause, nstem))


