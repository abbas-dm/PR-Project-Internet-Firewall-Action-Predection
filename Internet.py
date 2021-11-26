import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
