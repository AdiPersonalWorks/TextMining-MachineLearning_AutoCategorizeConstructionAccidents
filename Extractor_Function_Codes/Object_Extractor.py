"""

Code to extract all the objects from the osha file
Created on Tue Oct 18 19:23:38 2016
@author: TEAM

"""

import nltk
import csv
from nltk import word_tokenize
from nltk import pos_tag
from nltk.chunk import *
import pandas as pd

# Reading the osha file
osha_file = pd.read_csv("../Datasets/OSHA_CauseTitle.csv")

reader = osha_file.Title
final_list = []

for row in reader:
    row = row.lower() # Converting to lower case
    
	# POS Tagging
	pos = pos_tag(word_tokenize(row))
    
	# Chunking rules
    obj_gram = r"""
        obj: {<IN><NN><CC><NN>+}
             {<IN><NN><TO><NNS>}
             {<IN><NN><TO><NN>+}
             {<IN><NN>+}
             {<IN><JJ><NN>+}
             {<IN><JJ><VBG><NN>+}
             {<IN><JJ><NNS>}             
             {<IN><JJ>}
             {<IN><VBG*><NN>+}
             {<IN><VBG><JJ><NN>+}
             {<IN><VBG><JJ>}
             {<IN><VBG><VBG><NN>+}
             {<IN><VBG><VBN><NN>+}
             {<IN><VBG><CC><NN>+}
             {<WRB><JJ><NN>+}
             {<WRB><JJ>}
             {<WRB><NN><NNS>}
             {<WRB><NN>+}
             {<WRB><NNS>}
             {<WRB><VBG><NNS>}
             {<IN><DT><NN>+}
             {<JJ><NN>+}
             {<IN><NNS>}
             {<VBG><DT><NN>+}
             {<IN><CD><NNS>+}
             {<IN><VBN><NN>+}
             {<WRB><VBG><NN>+}
             {<IN><VBG><NNS>}
             {<TO><VBG><NN>+}
             {<TO><VB><NN>+}
             {<TO><NN>+}
             {<IN><DT><VBG><NN>+}
             {<VBG><VBG><NN>+}
             {<IN><VBG><RB><NN>+}
             {<TO><VB><NNS>}
             {<TO><DT><NN>+}
             {<VBP><VBG><NN>+}
             {<IN><JJR><NN>+}
             {<IN><VBG><RP><NN>+}
             {<IN><CD><NN><NNS>}
             {<IN><CD><NN>+}
     """
    
    test_chunker = nltk.RegexpParser(obj_gram)
    test_chunker_result = test_chunker.parse(pos)
    
    tree = test_chunker_result
    #print tree
    tot_word = []
    for a in tree:
        if type(a) is nltk.Tree and a.label() == 'obj':
            #print a
            for seg in a.leaves():
                if seg[1] == 'NN':
                    tot_word.append(seg[0])
    f_word = " ".join(x for x in tot_word)
    print(f_word)
    
    final_list.append(f_word)

# Writing the results to a file
with open("../Datasets/ObjectsExtracted.csv",'wb') as f:
    writer = csv.writer(f)
    writer.writerows(zip(reader,final_list))