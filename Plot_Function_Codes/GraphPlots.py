'''

File to plot statistics related to accidents
@author: Team

'''

import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

masia=pd.read_csv("../Datasets/OSHA_Predicted_Causes.csv")

causes = masia.Cause
causes = [x for x in causes if str(x) != 'nan']
word_counts = Counter(causes)

df = pd.DataFrame.from_dict(word_counts,orient='index')
ax = df.plot(kind = 'bar', legend = False,rot = 80)

for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
    
# Pie chart
df.plot.pie(subplots = True, legend = False, colors=['r', 'g', 'b', 'c','y'], autopct='%.2f')

plt.show()
