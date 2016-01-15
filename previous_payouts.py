
import pandas as pd


def previous_payout(df):
    idx = [i for i in xrange(len(df)) if (len(df.loc[i, 'previous_payouts']) == 0)]
    df['previous_payout'] = 1
    df.loc[idx, 'previous_payout'] = 0
    df.drop('previous_payouts', axis=1, inplace=True)
    return df