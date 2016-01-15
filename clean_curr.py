import pandas as pd

def currency_dummy(df):
    currency = pd.get_dummies(df['currency'])
    cols = currency.columns.tolist()
    cols = [col[0] for col in cols]
    currency.columns = cols
    df.drop('currency', axis=1, inplace=True)
    return df.join(currency)

def currency_dummy_point(df):
    currencies = ['A', 'C', 'E', 'G', 'M', 'N', 'U']
    df['A'] = 0
    df['C'] = 0
    df['E'] = 0
    df['G'] = 0
    df['M'] = 0
    df['N'] = 0
    df['U'] = 0
    for row_idx in df.index.values:
        for currency in currencies:
            if df.loc[row_idx, 'currency'][0] == currency:
                df.loc[row_idx, currency] = 1
    df.drop('currency', axis=1, inplace=True)
    return df


def clean_sale_duration(df):
    #make all negative values positive
    df['sale_duration'] = abs(df['sale_duration'])
    return df

def create_event_length(df):
    #calculate event_length
    df['event_length'] = abs(df['event_end'] - df['event_start'])
    df = df.drop(['event_start','event_end'], axis=1)
    return df

def currency_and_duration(df, point_predict=False):
    if point_predict:
        df = currency_dummy_point(df)
    else:
        df = currency_dummy(df)
    df = clean_sale_duration(df)
    df = create_event_length(df)
    return df

if __name__ == '__main__':
    df = currency_dummy(df)
    df = clean_sale_duration(df)
    df = event_length(df)