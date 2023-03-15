import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data from CSV file
data = pd.read_csv('../data/claim_reserving_data_updated.csv', nrows=10000)

# Split data into features and target variable
X = data.drop('TOTAL_INCURRED', axis=1)
y = data['TOTAL_INCURRED']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create SVM model
model = svm.SVC(kernel='linear')

# Train SVM model
model.fit(X_train, y_train)

# Predict target variable for test data
y_pred = model.predict(X_test)

# Evaluate model performance using accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)