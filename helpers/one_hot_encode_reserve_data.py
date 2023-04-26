import pandas as pd


def claims_reserve_ohe(claims_data):
    # Get a sense of what the data looks like
    print(claims_data.head())

    # let's have a look at the types
    # print(claims_data['INJURY_CAUSE'].unique())
    # print(claims_data['INJURY_CAUSE'].unique().size)
    #
    # print(claims_data['BODY_PART'].unique())
    # print(claims_data['BODY_PART'].unique().size)
    #
    # print(claims_data['NATURE_OF_INJURY'].unique())
    # print(claims_data['NATURE_OF_INJURY'].unique().size)

    # claims_data_dropped_columns = claims_data.drop(columns=['INJURY_STATE', 'JURISDICTION_STATE', 'LossDate',
    #                                                         'UpdatedDate', 'POLICY_NAICS_SECTOR', 'POLICY_NAICS_CODE'])

    # update for revised data file for SVM
    claims_data_dropped_columns = claims_data.drop(columns=['POLICY_NAICS_CODE'])

    # claims_data_dropped_columns = claims_data.dropna()

    print(claims_data_dropped_columns.head())

    one_hot_encoded = pd.get_dummies(claims_data_dropped_columns, columns=['INJURY_CAUSE', 'BODY_PART',
                                                                           'NATURE_OF_INJURY', 'CLAIM_TYPE',
                                                                           'OCCUPATION', 'CLASS_CODE'])
    print(one_hot_encoded.head())

    one_hot_encoded = one_hot_encoded.dropna()

    print(one_hot_encoded.isnull().values.any())

    return one_hot_encoded
