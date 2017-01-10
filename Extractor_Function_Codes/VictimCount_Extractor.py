'''
Code to extract the number of victims from the osha investigation summaries
'''

# Importing Libraries
import string
import io
import pandas as pd
import re
import csv

# Reading the titles from the OSHA file
data=pd.read_csv("../Datasets/OSHA_CauseTitle.csv")

countst=0
countmt=0

title=data.Title

reg1 = 'victims|employees|drivers|workers|operators|mechanics|owners'
reg2 = 'victim|employee|driver|worker|operator|mechanic|owner|\''
reg3 = '\swere\s|\sare\s|\stwo\s|\sthree\s|\sfour\s|\sfive\s|\ssix\s|\sseven\s|\seight\s|\snine\s|\sten\s|\seleven\s|\stwelve\s|\sthirteen\s'

creg1 = re.compile(reg1, re.IGNORECASE)
creg2 = re.compile(reg2, re.IGNORECASE)
creg3 = re.compile(reg3, re.IGNORECASE)

finallistt = []

for i in title:
    found = creg1.findall(i)
    if len(found) > 0:
        finallistt.append("Multiple")
        countmt=countmt+1
    else:
        found = creg2.findall(i)
        if len(found) > 0:
            finallistt.append("Single")
            countst=countst+1
        else:
            found = creg3.findall(i)
            if len(found) > 0:
                finallistt.append("Multiple")
                countmt=countmt+1
            else:
                finallistt.append("Single")
                countst=countst+1
   
# Writing the results
with open("../Datasets/Victims_Extracted.csv",'wb') as f:
    writer = csv.writer(f)
    writer.writerows(zip(title,finallistt))
    
print "Count single title =%d" % countst
print "Count Multiple title =%d" % countmt
