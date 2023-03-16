# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import helpers.one_hot_encode_reserve_data as ohe
from svm import run_model


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Read the CSV file into a pandas dataframe
    df = pd.read_csv('data/claim_reserving_data_updated.csv', nrows=10000)

    model_data = ohe.claims_reserve_ohe(df)

    run_model(model_data)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
