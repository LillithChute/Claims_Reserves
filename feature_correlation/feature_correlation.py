# Info from two websites used:
# http://shakedzy.xyz/dython/
# https://blog.knoldus.com/how-to-find-correlation-value-of-categorical-variables/#lets-find-the-correlation-of-categorical-variable 

# dython is a Python library that provides a set of functions for data exploration and visualization. 
# It is built on top of pandas, matplotlib, and other Python libraries.  
# Specifically designed for data analysis especially for machine learning and data science.

# WARNING - dython requires and WILL update the installed version of the following packages:
# numpy, pandas, matplotlib, seaborn, scipy, scikit-learn, and statsmodels
# Install on 3/28 using "python -m pip install dython" installed the following versions:
# contourpy-1.0.7 dython-0.7.3 matplotlib-3.7.1 pandas-1.5.3 psutil-5.9.4 scikit-plot-0.3.7 scipy-1.10.1 seaborn-0.12.2

import pandas as pd
from dython.nominal import associations
from dython.nominal import identify_nominal_columns

# Read the CSV file into a pandas dataframe
df = pd.read_csv('../data/Claim_Reserving_Data_UPDATED.csv', nrows=10000)

# test print to verify data is being read
print(df.head(5))

# identify categorical variables
# categorical_features=identify_nominal_columns(df)

# test print to verify categorical variables are being identified
# print(categorical_features)


# basic dython correlation matrix creation and display
# dython automatically identifies categorical variables in the process
# associations(df)

# generate the correlation matrix and save it to a file
complete_correlation= associations(df, filename= 'complete_correlation.png', figsize=(20,20))