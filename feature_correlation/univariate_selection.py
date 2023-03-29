import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2, f_classif


def univariate_f_statistic(data):

    # Fill missing values
    for column in data.columns:
        if column in ["POLICY_NAICS_SECTOR", "POLICY_NAICS_CODE"]:
            data[column].fillna(0, inplace=True)  # Fill missing values with 0
        elif data[column].dtype == np.number:
            data[column].fillna(data[column].median(skipna=True), inplace=True)  # For numeric columns
        else:
            data[column].fillna(data[column].mode(dropna=True).iloc[0], inplace=True)  # For categorical columns

    # Check for remaining NaN values
    print("Remaining NaN values in the dataset:", data.isna().sum().sum())

    print(data.head())
    print(data.dtypes)

    # Remove constant columns
    data = data.loc[:, data.apply(pd.Series.nunique) != 1]

    # Convert categorical columns to numerical
    for column in data.select_dtypes(include=['object']):
        data[column] = pd.factorize(data[column])[0]

    # Select the 10 best features
    X = data.drop("TOTAL_INCURRED", axis=1)
    y = data["TOTAL_INCURRED"]
    selector = SelectKBest(score_func=f_classif, k=10)
    selector.fit(X, y)

    # Get scores and feature names
    scores = selector.scores_
    features = X.columns

    # Visualize the 10 best features and their correlation scores
    best_features = pd.DataFrame({'Feature': features, 'Score': scores}).sort_values(by='Score', ascending=False).head(10)

    plt.figure(figsize=(10, 5))
    sns.barplot(x='Score', y='Feature', data=best_features)
    plt.title('Top 10 Features and Their Correlation Scores')
    plt.xlabel('Correlation Score')
    plt.ylabel('Feature')
    plt.show()


def univariate_pearson(data):

    # Fill missing values
    for column in data.columns:
        if column in ["POLICY_NAICS_SECTOR", "POLICY_NAICS_CODE"]:
            data[column].fillna(0, inplace=True)  # Fill missing values with 0
        elif data[column].dtype == np.number:
            data[column].fillna(data[column].median(skipna=True), inplace=True)  # For numeric columns
        else:
            data[column].fillna(data[column].mode(dropna=True).iloc[0], inplace=True)  # For categorical columns

    # Check for remaining NaN values
    print("Remaining NaN values in the dataset:", data.isna().sum().sum())

    print(data.head())
    print(data.dtypes)

    # Remove constant columns
    data = data.loc[:, data.apply(pd.Series.nunique) != 1]

    # Convert categorical columns to numerical
    for column in data.select_dtypes(include=['object']):
        data[column] = pd.factorize(data[column])[0]

    # Calculate Pearson correlation coefficients between the features and the target variable
    correlation_matrix = data.corr()
    correlation_scores = correlation_matrix["TOTAL_INCURRED"]

    # Visualize the top 10 features and their correlation coefficients
    best_features = correlation_scores.sort_values(ascending=False).head(
        11)  # Select 11 because TOTAL_INCURRED is one of them
    best_features = best_features.drop("TOTAL_INCURRED")  # Remove TOTAL_INCURRED from the list

    plt.figure(figsize=(10, 5))
    sns.barplot(x=best_features.values, y=best_features.index)
    plt.title('Top 10 Features and Their Correlation Coefficients')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Feature')
    plt.show()