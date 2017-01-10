'''
Activities Extractor using Stanford LP
@author - TEAM

'''

# Importing libraries
import os
import csv
import pandas as pd
from nltk.parse import stanford
from nltk import sent_tokenize
from nltk.parse.stanford import StanfordDependencyParser

oshasum = pd.read_csv('../Datasets/OSHA_Summary.csv')
summ = oshasum.Summary
count = 1

# Setting the environment variables
# If required please set the JAVA environment as well
os.environ['STANFORD_PARSER'] = '../Parsers/Stanford Parsers/jars'
os.environ['STANFORD_MODELS'] = '../Parsers/Stanford Parsers/jars'
dependencyParser = StanfordDependencyParser()
parser = stanford.StanfordParser(model_path="../Parsers/Stanford Parsers/jars/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

# Some variable declarations
finsent = []
all_acts = []

try:
    for sin_summ in summ:       

        reqsent = sent_tokenize(sin_summ)
        sin_summ = reqsent[0]
        
        sentences = parser.raw_parse(sin_summ)

        # Dependency parsing to extract universal dependencies
        result = dependencyParser.raw_parse(sin_summ)
        dep = result.next()
        trips = list(dep.triples())

        # Finding the dobj dependency
        req_activity = []
        for t in trips:
            if t[1] == 'dobj':
                req_activity = t[0][0] + '_' + t[2][0]
                break
        all_acts.append(req_activity)

        # Just here
        for line in sentences:
            for sentence in line:
                print(count)
                count = count+1
    ##       sentence.draw()

        sent = []
        for i in line.subtrees():
            if i.label() == 'VP':
                sent.append(i.leaves())

        try:
            tmp_sent = sent[0]
        except:
            tmp_sent  = ''
            finsent.append(tmp_sent)
            continue
        
        finsent.append(tmp_sent)

    # Final VP set from the summary
    fin_sumVPset = []
    for sin_finset in finsent:
        tmp_finset = " ".join(w for w in sin_finset)
        fin_sumVPset.append(tmp_finset)
    ##print(fin_sumVPset)

    # Final Acts set from summary
    fin_actsset = []
    for sin_act in all_acts:
        tmp_act = "".join(w for w in sin_act)
        fin_actsset.append(tmp_act)
    ##print(fin_actsset)

    with open('Activities_Extracted.csv','wb') as f:
        wri = csv.writer(f)
        wri.writerows(zip(summ,fin_actsset,fin_sumVPset))
except:
    # Final VP set from the summary
    fin_sumVPset = []
    for sin_finset in finsent:
        tmp_finset = " ".join(w for w in sin_finset)
        fin_sumVPset.append(tmp_finset)
    ##print(fin_sumVPset)

    # Final Acts set from summary
    fin_actsset = []
    for sin_act in all_acts:
        tmp_act = "".join(w for w in sin_act)
        fin_actsset.append(tmp_act)
    ##print(fin_actsset)

    with open('OSHA_Activities_StanfordLP.csv','wb') as f:
        wri = csv.writer(f)
        wri.writerows(zip(summ,fin_actsset,fin_sumVPset))  