import pandas as pd
from sklearn.decomposition import PCA

# Load data from CSV file
data = pd.read_csv('../data/claim_reserving_data_updated.csv', nrows=10000)

# Separate features from the target variable (if applicable)
X = data.drop('TOTAL_INCURRED', axis=1)

# Create PCA object with desired number of components
n_components = 2  # Change this to the desired number of components
pca = PCA(n_components=n_components)

# Fit and transform the data to the PCA space
X_pca = pca.fit_transform(X)

# Print explained variance ratio of the components
print("Explained variance ratio:", pca.explained_variance_ratio_)

# Print the singular values of the components
print("Singular values:", pca.singular_values_)
