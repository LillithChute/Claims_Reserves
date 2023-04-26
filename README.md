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


## univariate_selection.py
This file does a F-Statistic and Pearson analysis to the dataframe passed in.  The methods are called in main.py.

### F-Statistic and Pearson Correlation Coefficient
#### F-Statistic
The F-statistic is a statistical measure used to assess the significance of the relationship between variables in a linear regression model. It compares the explained variance by the model to the unexplained variance (residuals), and helps determine whether the variables included in the model provide useful information for predicting the outcome. A higher F-statistic value indicates a stronger relationship between the variables, while a lower value suggests that the model is not significantly better than simply using the mean of the dependent variable as a predictor.

In other words, the F-statistic tests the null hypothesis that all regression coefficients are equal to zero, meaning that none of the independent variables have a significant effect on the dependent variable. If the F-statistic is significantly large, we reject the null hypothesis, concluding that at least one of the independent variables has a significant relationship with the dependent variable.

#### Pearson Correlation Coefficient
The Pearson correlation coefficient, often denoted as 'r' or 'ρ', is a measure of the strength and direction of the linear relationship between two continuous variables. It ranges from -1 to 1, with -1 indicating a strong negative correlation, 1 indicating a strong positive correlation, and 0 suggesting no correlation between the variables.

The Pearson correlation coefficient is useful for identifying potential relationships between variables and determining which variables may be worth including in a linear regression model. However, it is important to remember that correlation does not imply causation, and other factors may be influencing the observed relationship.

In summary, the F-statistic helps assess the overall significance of a linear regression model, while the Pearson correlation coefficient quantifies the strength and direction of the linear relationship between two continuous variables. Both are essential tools for understanding relationships between variables in statistical analyses.

## data_visualization_functions.py
This file contains two methods that generate a bar graph using the injury_cause.csv file.  It calculates the top ten injury causes using the original aggregated cost amount (not normalized to today's dollars).  The second visualization is using the normalized dollars.  I wanted to see if the order of highest expense to lowest stayed with the same type of injury order regardless and how expensive it was.

These visualizations are run via the _main.py_ file.

## Dimensionality Reduction directory
Had we had data that was usable, we intended to use PCA and UMAP techniques to do further feature correlation and dimensionality reduction.  We expected the data to result in high dimensionality once we had everything we needed.  Thus, we created some boilerplate code to start with.

