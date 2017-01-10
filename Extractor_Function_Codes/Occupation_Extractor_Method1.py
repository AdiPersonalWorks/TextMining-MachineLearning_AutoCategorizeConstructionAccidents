"""
Code to extract the different occupations from the osha file
"""

import nltk
import csv
from nltk import word_tokenize
from nltk import pos_tag
from nltk.chunk import *

# Chunking rules
occupation_grammar = r"""
  Occ: {<NN><VBZ><VBN><IN>}
       {<NN><NNS><VBP>}
       {<NN><POS>}
       {<NN><VBN>}
       {<NN><NNS><IN><VBN>}
       {<NN><NNS><NNS><IN><VBG>}
       {<NN><NNS><JJ>}
       {<NN><NNS><IN><JJ>}
       {<NN><NNS><IN>}
       {<NN><NNS><NNS><IN><JJ>}
       {<NN><NNS><CC><JJ>}
       {<NN><NNS><NNS><WRB>}
       {<NN><VBD>}
       {<NN><NN><IN>}
       {<NN><NNS><WRB>}
"""

import pandas as pd
# Read the required file
osha_file = pd.read_csv("../Datasets/OSHA_CauseTitle.csv")

titles = osha_file.Title
final_list = []

for s in titles:
    s = s.lower()
    
    pos = pos_tag(word_tokenize(s)) 
    
    test_chunker = nltk.RegexpParser(occupation_grammar)
    test_chunker_result = test_chunker.parse(pos)
    
    tot_word = []
    tree = test_chunker_result
    for a in tree:
        if type(a) is nltk.Tree and a.label() == 'Occ':
            for seg in a.leaves():
                if seg[1] == 'NN':
                    tot_word.append(seg[0])
    f_word = " ".join(x for x in tot_word)
    print(f_word)
    
    final_list.append(f_word)

# Writing the results to a file
with open("../Datasets/OccupationsExtracted_Method1.csv",'wb') as f:
    writer = csv.writer(f)
    writer.writerows(zip(titles,final_list))                