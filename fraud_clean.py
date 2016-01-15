import pandas as pd


def create_fraud_col(df):
    df['fraud'] = df['acct_type'].str.contains('fraud').astype(int)
    return df.drop('acct_type', axis=1)