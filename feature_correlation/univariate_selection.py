import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression

# Read the CSV file into a pandas dataframe
df = pd.read_csv('../data/claim_reserving_data_updated.csv', nrows=10000)

# Separate the features and target variable
X = df.iloc[:, :-1]  # Select all columns except the last one
y = df.iloc[:, -1]   # Select the last column

# Perform Univariate Selection
selector = SelectKBest(score_func=f_regression, k=5)
X_new = selector.fit_transform(X, y)

# Print the selected features
print(X.columns[selector.get_support()])