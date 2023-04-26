import pandas as pd

# Read the CSV file into a pandas dataframe
df = pd.read_csv('../data/claim_reserving_data_updated.csv', nrows=10000)

# Generate the correlation matrix
corr_matrix = df.corr()

# Print the correlation matrix
print(corr_matrix)