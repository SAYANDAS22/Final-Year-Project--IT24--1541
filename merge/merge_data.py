import pandas as pd

# Read in the CSV files
df1 = pd.read_csv('Gdp.csv')
df2 = pd.read_csv('Population.csv') 
df3 = pd.read_csv('Historical.csv')

# Merge df1 and df2 on the 'STATE' column
df_merged = df1.merge(df2, on='STATE')

# Merge the result with df3 on the 'YEAR' column 
df_merged = df_merged.merge(df3, on='YEAR')

# Display the merged DataFrame
print(df_merged)