import pandas as pd

df = pd.read_csv("data/dataset_phishing.csv")

print(df.shape)          # How many rows and columns?
print(df.columns.tolist())  # What are the column names?
print(df.head())         # What does the first 5 rows look like?
print(df['status'].value_counts())  # How many phishing vs legitimate?