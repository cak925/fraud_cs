import pandas as pd
import numpy as np


def clean_venue_state(df, min_num=20, min_fraud=0.1, return_state_percent=False, point_predict=False):
    # Converts all 'venue_state' entries to uppercase
    df['venue_state'] = df['venue_state'].str.upper()

    # Subset the dataframe to look at only the entries from the US
    US_df = df[df['venue_country'] == 'US']
    US_df['venue_state'] = US_df['venue_state'].str.replace(r'[0-9]', '')
    US_df['venue_state'] = state_replace(US_df['venue_state'])

    canada = 'ONTARIO|CANADA|QUEBEC|ON.|AB|QC|ON|BC'
    australia = 'QLD|WESTERN AUSTRALIA|AUSTRALIA'
    jamaica = 'JAMAICA'
    thailand = 'THAILAND'
    england = 'NEW SOUTH WALES'

    # Some of the values in the 'venue_state' field are actually the country.
    # Reset each row with any of the values contained in the above strings to
    # the appropriate country code
    US_df.loc[US_df['venue_state'].str.contains(
        canada), 'venue_country'] = 'CA'
    US_df.loc[US_df['venue_state'].str.contains(
        australia), 'venue_country'] = 'AU'
    US_df.loc[US_df['venue_state'].str.contains(
        jamaica), 'venue_country'] = 'JM'
    US_df.loc[US_df['venue_state'].str.contains(
        thailand), 'venue_country'] = 'TH'
    US_df.loc[US_df['venue_state'].str.contains(
        england), 'venue_country'] = 'GB'

    # Merge US_df back into the original dataframe and resplit on the United
    # States so as to remove states which aren't in the US
    df.loc[US_df.index, :] = US_df
    US_df = df[df['venue_country'] == 'US']

    # Limit each 'venue_state' to two characters
    US_df['venue_state'] = US_df['venue_state'].apply(lambda x: x[:2])
    states = 'AL|AK|AZ|AR|CA|CO|CT|DE|DC|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY'

    # Set any state which isn't contained in the above states list to be nan
    US_df.loc[
        ~(US_df['venue_state'].str.contains(states)), 'venue_state'] = np.nan

    # Remerge US_df into the orginal df
    df.loc[US_df.index, :] = US_df

    # Make dummies from the states based off a certain number of records as
    # well as a minimum percentage of fraud cases
    if return_state_percent:
        df, state, percent_fraud, count = make_state_dummies(
            df, min_num=min_num, min_fraud=min_fraud, return_state_percent=True)
        return df, state, percent_fraud, count
    elif not point_predict:
        df = make_state_dummies(df, min_num=20, min_fraud=0.1)
        return df
    else:
        df = make_state_dummies_point(df)
        return df


def state_replace(series):
    series = series.str.replace('MASSACHUSETTS', 'MA')
    series = series.str.replace('PENNSYLVANIA', 'PA')
    series = series.str.replace('ALAKSA', 'AK')
    series = series.str.replace('CALIFORNIA|TIBURON', 'CA')
    series = series.str.replace('HAWAII', 'HI')
    series = series.str.replace('OHIO', 'OH')
    series = series.str.replace('VIRGINA', 'VA')
    series = series.str.replace('MICHIGAN', 'MI')
    series = series.str.replace('D.C.|DISTRICT OF COLUMBIA', 'DC')
    series = series.str.replace('SOUTH CAROLINA', 'SC')
    series = series.str.replace('FLORIDA', 'FL')
    series = series.str.replace('TEXAS', 'TX')
    series = series.str.replace('KENTUCKY', 'KY')
    series = series.str.replace('VERMONT', 'VT')
    series = series.str.replace('NORTH CAROLINA', 'NC')
    series = series.str.replace('NEW JERSEY', 'NJ')
    series = series.str.replace('VIRGINIA', 'VA')
    return series


def state_percentages(df, min_num=20, min_fraud=0.1):
    states = df['venue_state'].unique()
    percent_fraud = np.array(
        [np.mean(df.loc[df['venue_state'] == state, 'fraud']) for state in states])
    counts = np.array([len(df[df['venue_state'] == state])
                       for state in states])
    idx = np.argsort(percent_fraud)[::-1]
    states, percent_fraud, counts = states[
        idx], percent_fraud[idx], counts[idx]
    idx = np.where(counts >= min_num)
    states, percent_fraud, counts = states[
        idx], percent_fraud[idx], counts[idx]
    idx = np.where(percent_fraud >= min_fraud)
    return states[idx], percent_fraud[idx], counts[idx]


def make_state_dummies(df, min_num, min_fraud, return_state_percent=False):
    states, percent_fraud, counts = state_percentages(
        df, min_num=min_num, min_fraud=min_fraud)
    dummies = pd.get_dummies(df['venue_state'])[states]
    cols = dummies.columns.tolist()
    cols = ['state_' + col.replace(' ', '_') for col in cols]
    dummies.columns = cols
    if return_state_percent:
        return df.join(dummies), states, percent_fraud, counts
    else:
        df.drop('venue_state', axis=1, inplace=True)
        return df.join(dummies)

def make_state_dummies_point(df):
    states = ['GREATER_LONDON', 'FL', 'LONDON', 'GT_LON', 'DE', 'BIRMINGHAM', 'PA', 'NV', 'NH', 'GA', 'ENGLAND']

    for state in states:
        col = 'state_' + state
        df[col] = 0

    for row_idx in df.index.values:
        for state in states:
            if df.loc[row_idx, 'venue_state'] == state:
                col = 'state_' + state
                df.loc[row_idx, col] = 1
    df.drop('venue_state', axis=1, inplace=True)
    return df