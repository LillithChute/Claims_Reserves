import matplotlib.pyplot as plt
import pandas as pd
import umap

# Load data from CSV file
data = pd.read_csv('../data/claim_reserving_data_updated.csv', nrows=10000)

# Separate features from the target variable (if applicable)
X = data.drop('TOTAL_INCURRED', axis=1)

# Create UMAP object with desired number of components and parameters
n_components = 2  # Change this to the desired number of components
umap_obj = umap.UMAP(n_components=n_components, n_neighbors=5, min_dist=0.3)

# Fit and transform the data to the UMAP space
X_umap = umap_obj.fit_transform(X)

# Plot the UMAP embedding using matplotlib or other libraries
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=data['TOTAL_INCURRED'])
plt.show()
