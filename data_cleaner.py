import pandas as pd 
#from eda import load_data

def date_clean(df):
    ''' Takes in the dataframe and converts the epoch time to object type
    datetime for pandas dataframe'''
    df['event_end'] = df['event_end'] // 100
    df['event_start'] = df['event_start'] // 100
    df.user_created = pd.to_datetime(df.user_created, unit = 's')
    df.approx_payout_date = pd.to_datetime(df.approx_payout_date, unit = 's')
    df.event_created = pd.to_datetime(df.event_created, unit = 's')
    df.event_end = pd.to_datetime(df.event_end, unit = 's')
    df.event_start = pd.to_datetime(df.event_start, unit = 's')
    df.event_published = pd.to_datetime(df.event_start, unit = 's')
    return df

if __name__ == '__main__':
    #df = load_data()
    df = date_clean(df)