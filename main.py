# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import helpers.one_hot_encode_reserve_data as ohe
from data_visualization_functions import injury_cause_type_count, injury_cause_type_by_normalized_incurred
from svm import run_model
from univariate_selection import univariate_f_statistic, univariate_pearson


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def run_model(data):
    model_data = ohe.claims_reserve_ohe(df)

    run_model(model_data)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Read the CSV file into a pandas dataframe
    df = pd.read_csv('data/claim_reserving_data_updated.csv', nrows=10000)

    injury_cause_df = pd.read_csv('data/Injury_Cause.csv')

    injury_cause_type_count(injury_cause_df)

    injury_cause_type_by_normalized_incurred(injury_cause_df)

    univariate_f_statistic(df)

    univariate_pearson(df)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
