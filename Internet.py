import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Reading the Dataset .csv file and analyzing the dataset
df = pd.read_csv("Internet_Project_Data.csv")

# Checking data is correctly read or not
print(df.head())

# Checking the data information to find is there any missing data
print(df.info())

# Finding the shape of dataset
print(df.shape)

# Finding  the percentage of Action/Target and Plotting the count of each Action/Target in the dataset given
print('Percentages: of each Action')
print(df.Action.value_counts(normalize=True))

sns.countplot(df['Action'],label="Count")
plt.savefig('Action_Count.png')




num_features = ['Bytes', 'Bytes Sent', 'Bytes Received', 'Packets', 'Elapsed Time (sec)', 'pkts_sent', 'pkts_received']

# Understanding the range of values in numerical features with the help of box plots
# This function is to get input for box plot
def num_trafo(x):
    return np.log10(1+x)

for f in num_features:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11,7), sharex=True)
    ax1.hist(num_trafo(df[f]), bins=20)
    ax1.grid()
    ax1.set_title('Feature: ' + f + ' - trafo [log_10(1+x)]')
    ax2.boxplot(num_trafo(df[f]), vert=False)
    ax2.grid()   
    ax2.set_title('Feature: ' + f + ' - trafo [log_10(1+x)]')
    plt.savefig(f)

# Understanding the range of values in Categorical Features
cat_features = ['Source Port', 'Destination Port', 'NAT Source Port', 'NAT Destination Port']

for f in cat_features:
    print('Feature:', f)
    print(df[f].value_counts()[0:10])
    print()

# Understanding the speard of values of Categorical Features with respect to Actions
# This is done by considering only some of top values in each feature
for f in cat_features:
    top10_levels = df[f].value_counts()[0:10].index.to_list()
    df_temp = df[df[f].isin(top10_levels)]
    ctab = pd.crosstab(df_temp.Action, df_temp[f])
    print('Feature:' + f + ' - Top 10 levels only')
    plt.figure(figsize=(12,5))
    sns.heatmap(ctab, annot=True, fmt='d', cmap='Blues', linecolor='black', linewidths=0.1)
    plt.savefig(f)


# Plotting the scatter plots between Source port and Destination Port
# Against each Action
xx = 'Source Port'
yy = 'Destination Port'

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10,10))

df_temp = df[df.Action=='Allow']
axs[0,0].scatter(df_temp[xx], df_temp[yy], alpha=0.05)
axs[0,0].set_title('Action = allow')
axs[0,0].set_xlabel(xx)
axs[0,0].set_ylabel(yy)
axs[0,0].grid()

df_temp = df[df.Action=='Deny']
axs[0,1].scatter(df_temp[xx], df_temp[yy], alpha=0.05)
axs[0,1].set_title('Action = deny')
axs[0,1].set_xlabel(xx)
axs[0,1].set_ylabel(yy)
axs[0,1].grid()

df_temp = df[df.Action=='Drop']
axs[1,0].scatter(df_temp[xx], df_temp[yy], alpha=0.5)
axs[1,0].set_title('Action = drop')
axs[1,0].set_xlabel(xx)
axs[1,0].set_ylabel(yy)
axs[1,0].grid()

df_temp = df[df.Action=='Reset-both']
axs[1,1].scatter(df_temp[xx], df_temp[yy], alpha=0.5)
axs[1,1].set_title('Action = reset-both')
axs[1,1].set_xlabel(xx)
axs[1,1].set_ylabel(yy)
axs[1,1].grid()

plt.savefig(xx+'_'+yy)

# Plotting the scatter plots between NAT Source port and NAT Destination Port
# Against each Action
xx = 'NAT Source Port'
yy = 'NAT Destination Port'

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(10,10))

df_temp = df[df.Action=='Allow']
axs[0,0].scatter(df_temp[xx], df_temp[yy], alpha=0.05)
axs[0,0].set_title('Action = allow')
axs[0,0].set_xlabel(xx)
axs[0,0].set_ylabel(yy)
axs[0,0].grid()

df_temp = df[df.Action=='Deny']
axs[0,1].scatter(df_temp[xx], df_temp[yy], alpha=0.5)
axs[0,1].set_title('Action = deny')
axs[0,1].set_xlabel(xx)
axs[0,1].set_ylabel(yy)
axs[0,1].grid()

df_temp = df[df.Action=='Drop']
axs[1,0].scatter(df_temp[xx], df_temp[yy], alpha=0.5)
axs[1,0].set_title('Action = drop')
axs[1,0].set_xlabel(xx)
axs[1,0].set_ylabel(yy)
axs[1,0].grid()

df_temp = df[df.Action=='Reset-both']
axs[1,1].scatter(df_temp[xx], df_temp[yy], alpha=0.5)
axs[1,1].set_title('Action = reset-both')
axs[1,1].set_xlabel(xx)
axs[1,1].set_ylabel(yy)
axs[1,1].grid()

plt.savefig(xx+'_'+yy)
#----------------------------------------------------------------------------
