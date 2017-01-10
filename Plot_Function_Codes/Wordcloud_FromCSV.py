'''
Code to plot a wordcloud from csv files
'''

from os import path
from wordcloud import WordCloud
import pandas as pd

d = path.dirname(__file__)

# Read the whole text.
occs = pd.read_csv('../Datasets/Occupations_Extracted_Final.csv')
all_occ = occs.OCC
freq = occs.FREQ

tuples = [tuple(x) for x in zip(all_occ,freq)]

wordcloud = WordCloud().generate_from_frequencies(tuples)

import matplotlib.pyplot as plt
plt.imshow(wordcloud)
plt.axis("off")
plt.show()