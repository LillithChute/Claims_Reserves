# Claims_Reserves
This project is the semester project for Northeastern University CS5100 Foundations of Artificial Intelligence.  It focuses on feature engineering and dimensionality reduction techniques to generate good data for ingestion into a machine learning model that will predict reserve amounts for individual claims.

# Executable Files
There are several files that can be run in the project.  No files take any arguments and can be either run from a terminal window or by clicking the run command in PyCharm.

## main.py
This file performs some analysis of the data and creates corresponding visualizations.

## svm.py
This runs a simple regression Support Vector Machine on an Indemnity Only claims file, containing only the top 9 correlated features from the starting data.  Requires pandas, numpy, and sklearn modules to be installed.  It runs the SVM and prints some analysis on the accuracy of the results to the terminal including the: R-Squared value, Average Coefficients, and Root Mean Squared Error. It then performs a Cross Validation on the model and prints the resulting score value generated.

## feature_correlation.py
This file utilizes the dython module to perform a correlation analysis on all the features in a dataset against each other.  The result on the files used here is a 29x29 matrix giving the correlation of all features to each other.  Here separate files for Medical Only and Indemnity claims were used with the data already cleaned and split.  The file currently uses all of the Indemnity file (~73,000 lines) and 100,000 lines of the Medical Only data so the time to run will vary depending on the system running it.  The generated charts are displayed after generated and saved to feature_correlation subdirectory.



