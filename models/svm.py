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

def run_model(data):
    # Split data into features and target variable
    X = data.drop('TOTAL_INCURRED', axis=1)
    y = data['TOTAL_INCURRED']

    # Split data into training and testing sets
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
