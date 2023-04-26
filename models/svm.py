import pandas as pd
import numpy as np
from sklearn import svm, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt


# add function to this file to resolve issues with imports
def claims_reserve_ohe(claims_data):

    claims_data_dropped_columns = claims_data.drop(columns=['POLICY_NAICS_CODE'])

    one_hot_encoded = pd.get_dummies(claims_data_dropped_columns, columns=['INJURY_CAUSE', 'BODY_PART',
                                                                           'NATURE_OF_INJURY',
                                                                           'OCCUPATION', 'CLASS_CODE'])

    one_hot_encoded = one_hot_encoded.dropna()

    return one_hot_encoded

# function to create SVM model and evaluate it's results
def run_model(data):
    # Split data into features and target variable, which is the adjsuted incurred amount
    X = data.drop('ADJUSTED_INCURRED', axis=1)
    y = data['ADJUSTED_INCURRED']

    # Split data into training and testing sets, 70% training and 30% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    lr = linear_model.LinearRegression()
    print("Fitting")
    lr_model = lr.fit(X, y)

    print("predict")
    y_pred = lr_model.predict(X)
    print("score")
    lr_r2 = r2_score(y, y_pred)
    print("R squared: " + str(lr_r2))
    print("Average Coefficients: ", (abs(lr_model.coef_).mean()))
    print("Root Mean Squared Error: ", sqrt(mean_squared_error(y, y_pred)))
    print("Cross Val")
    from sklearn.model_selection import cross_val_score
    print(np.mean(cross_val_score(lr_model, X, y, n_jobs=1, cv=5)))
    print("Finished")

    # # Create SVM model
    # model = svm.SVR(kernel='rbf')
    #
    # # Train SVM model
    # model.fit(X_train, y_train)
    #
    # # Predict target variable for test data
    # y_pred = model.predict(X_test)
    #
    # # Evaluate model performance using accuracy score
    # accuracy = accuracy_score(y_test, y_pred)
    # print("Accuracy:", accuracy)

if __name__ == '__main__':
    # Read the CSV file into a pandas dataframe
    df = pd.read_csv('../data/Indemnity_Data_FINAL.csv', nrows=10000, encoding='unicode_escape')

    model_data = claims_reserve_ohe(df)

    run_model(model_data)
