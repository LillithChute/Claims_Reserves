# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
import helpers.one_hot_encode_reserve_data as ohe
from visualizations.data_visualization_functions import injury_cause_type_count, injury_cause_type_by_normalized_incurred
from models.svm import run_model
from feature_correlation_work.univariate_selection import univariate_f_statistic, univariate_pearson


def run_the_model(df):
    model_data = ohe.claims_reserve_ohe(df)

    run_model(model_data)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Read the CSV file into a pandas dataframe
    df_original = pd.read_csv('data/claim_reserving_data_updated.csv', encoding='unicode_escape', low_memory=False)

    df_indemnity = pd.read_csv('data/Indemnity_Claim_Reserving_Data_CLEANED.csv', encoding='unicode_escape'
                               , low_memory=False)

    df_medical_only = pd.read_csv('data/Medical_Only_Claim_Reserving_Data_CLEANED.csv', encoding='unicode_escape'
                                  , low_memory=False)

    injury_cause_df = pd.read_csv('data/Injury_Cause.csv')

    injury_cause_type_count(injury_cause_df)

    injury_cause_type_by_normalized_incurred(injury_cause_df)

    univariate_f_statistic(df_original)

    univariate_pearson(df_original)

    univariate_f_statistic(df_indemnity)

    univariate_pearson(df_indemnity)

    univariate_f_statistic(df_medical_only)

    univariate_pearson(df_medical_only)
