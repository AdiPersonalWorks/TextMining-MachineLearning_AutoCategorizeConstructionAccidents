'''
Word Frequency Counter code
'''

from collections import Counter
import csv

file = open(r"../Datasets/OccupationsForWordcloud.txt", "r")

wordcount = Counter(file.read().split())

with open("../Datasets/OccWordCount.csv",'a') as f:
    for item in wordcount.items():
        writer = csv.writer(f)
        writer.writerows(zip([item[0]],[str(item[1])]))
        print("{}\t{}".format(*item))
