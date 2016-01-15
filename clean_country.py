import numpy as np
import pandas as pd

def calculate_country_basics(df):
    ''' Calculate the number of unique countries. Return the countries
    and the length of the dataframe. '''
    unique_countries = df.country.unique()
    df_length = float(len(df))
    return unique_countries, df_length

def country_percentage(df):
    ''' Return a tuple of the country, fraction of fraud, and total
    number of appearances of that country in the dataset. '''
    unique_countries, df_length = calculate_country_basics(df)
    f = [(unique_country,
        np.sum(df.fraud.values[np.where(df.country == unique_country)])
        .astype(float) / np.sum(df.fraud.values),
        (df.country == unique_country).sum() / df_length)
        for unique_country in unique_countries]
    return np.array(f)

def make_country_dummies(df, lower_bound = 20):
    ''' Make dummies of the countries that have at least 20 appearances
    in the dataset. '''
    unique_countries, df_length = calculate_country_basics(df)
    country_frac = country_percentage(df)
    cutoff_frac = lower_bound / df_length
    countries_worth_making_dummies = unique_countries[country_frac[:, 2] >
                                        cutoff_frac]
    country_dummies = pd.get_dummies(df.country)[countries_worth_making_dummies]
    country_dummies['country_not_specified'] = country_dummies['']
    country_dummies.drop('', axis = 1, inplace = True)
    cols = country_dummies.columns.tolist()
    cols[-1] = 'other'
    cols = ['country_' + col for col in cols]
    country_dummies.columns = cols
    df = df.join(country_dummies)
    df.drop(['country', 'venue_country'], axis = 1, inplace = True)
    return df

def make_country_dummies_point(df):
    countries = ['US', 'IE', 'FR', 'CA', 'GB', 'AU', 'ES', 'NL', 'DE', 'VN', 'NZ', 'PK', 'MA', 'A1']

    for country in countries:
        col = 'country_' + country
        df[col] = 0
    df['country_other'] = 0

    for row_idx in df.index.values:
        set_val = False
        for country in countries:
            if df.loc[row_idx, 'country'] == country:
                col = 'country_' + country
                df.loc[row_idx, col] = 1
                set_val = True
        if not set_val:
            df.loc[row_idx, 'country_other'] = 1
    df.drop(['country', 'venue_country'], axis=1, inplace=True)
    return df