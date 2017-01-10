'''
Code to find the activity from the osha file using regular expressions
'''

# Importing required libraries
import re
import csv
from xlrd import open_workbook

# Read the required workbook
wb = open_workbook('../Datasets/osha_original.xlsx')
col=2
col_value=[]
activity=[]

for s in wb.sheets():
    for row in range(0, s.nrows):
            value  = (s.cell(row,col).value)
            col_value.append((value))
    
for rows in col_value:
    if "was " in rows:
        find_was=rows.index("was ")
        string_after_was = rows[find_was:]
        end_index=re.search( '\. '+r'[A-Z]',string_after_was)
        if end_index is not None:
            end_index=end_index.start()
            activity.append(string_after_was[:end_index])
        else:
            end_index=re.search("\. ",string_after_was)
            if end_index is not None:
                end_index = end_index.start()
                activity.append(string_after_was[:end_index])
            else:
                end_index = len(string_after_was)
                activity.append(string_after_was[:end_index])
    else:
        activity.append("Activity not found")

# Writing the results to a file
with open("../Datasets/OSHA_RegEx_Activities.csv",'wb') as f:
    writer = csv.writer(f)
    writer.writerows(zip(col_value,activity))
