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

# run a complete correlation matrix on the Medical Only claims dataframe on the first 100000 rows
# The 100000 row limit is to ensure the program runs in a reasonable amount of time on all systems

# first read in the Medical Only claims data into a dataframe
df = pd.read_csv('../data/Medical_Only_Claim_Reserving_Data_CLEANED.csv', nrows=100000, encoding='unicode_escape')

# manually identify categorical variables
categorical_features=['POLICY_NAICS_SECTOR', 'POLICY_NAICS_CODE', 'POLICY_GOVERNING_CLASS', 'CLAIM_TYPE', 'INJURY_CAUSE', 
                      'BODY_PART', 'NATURE_OF_INJURY', 'CLASS_CODE','INJURY_STATE', 'JURISDICTION_STATE', 'OCCUPATION']

# convert LossDate to a datetime object in the dataframe
df['LossDate'] = pd.to_datetime(df['LossDate'], format= '%m/%d/%Y')

# generate the correlation matrix and save it to a file
# complete_correlation= associations(df, nominal_columns=categorical_features, filename= 'medical_all_correlation.png', figsize=(20,20))
medical_correlation= associations(df, nominal_columns=categorical_features, filename= 'medical_all_correlation.png', title='Medical Only Correlation Matrix - 100k rows')

# In the correlation matrix, SV indicates that the feature has only a single value in the first 100000 rows

# we could check the single value features on a larger sample of the data to see if there is a correlation with the target variable
# but for now we will just remove the single value features from the data when running the SVM model


# run a complete correlation matrix on the Indemnity claims dataframe of all the data

# first read in the Indemnity claims data into the same dataframe object
df = pd.read_csv('../data/Indemnity_Claim_Reserving_Data_CLEANED.csv', encoding='unicode_escape')

# manually identify categorical variables UNCHANGED
categorical_features=['POLICY_NAICS_SECTOR', 'POLICY_NAICS_CODE', 'POLICY_GOVERNING_CLASS', 'CLAIM_TYPE', 'INJURY_CAUSE', 
                      'BODY_PART', 'NATURE_OF_INJURY', 'CLASS_CODE','INJURY_STATE', 'JURISDICTION_STATE', 'OCCUPATION']

# convert LossDate to a datetime object in the dataframe
df['LossDate'] = pd.to_datetime(df['LossDate'], format= '%m/%d/%Y')

# generate the correlation matrix and save it to a file
# complete_correlation= associations(df, nominal_columns=categorical_features, filename= 'indemnity_all_correlation.png', figsize=(20,20))
indemnity_correlation= associations(df, nominal_columns=categorical_features, filename= 'indemnity_all_correlation.png', title='Indemnity Correlation Matrix - 73k rows')



# print message to verify the files were generated successfully
print("Files generated successfully")