
# Kappa Measure
"""

from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

x = ['1- Approval' , '2- Code' ,
     '3-Design' , '4- Functional',
    '5- feedback' , '6-QA']
ratings = [
    [10 , 12], #1- Approval or rejection comments ,
    [6 , 6], #2- Code review comments ,
    [10 , 7], #3- Design review comments ,
    [10 , 8], #4- Functional review comments,
    [7 , 8], #5- General feedback comments ,
    [7 , 9] #6- Quality assurance (QA) comments
]


y1 = [10,6,10,10,7,7]

y2 = [12,6,7,8,8,9]

df = pd.DataFrame(ratings , columns = ['rater1', 'rater2'])
kappa = cohen_kappa_score(df['rater1'], df['rater2'], weights = 'linear')



# Set the width of each bar
bar_width = 0.1

# Create the figure and axes objects
fig, ax = plt.subplots()

# Create the first set of bars
ax.bar(x, y1, bar_width, label='Rater 1')

# Shift the x positions of the second set of bars
x2 = np.arange(len(x)) + bar_width

# Create the second set of bars
ax.bar(x2, y2, bar_width, label='Rater 2')


# Add labels, title, and legend
ax.set_xlabel('Categories')
ax.set_ylabel('Values')
ax.set_title('Bar Chart Example')
ax.legend()

# Show the chart
#plt.show()



print(f"The kappa measure is: {kappa}")

#plt.bar(kappa_df['Raters'] , kappa_df['Kappa'])
plt.ylim([0,20])
plt.ylabel('cohen_kappa_score')
plt.title('Inter-rater')


#plt.show()





