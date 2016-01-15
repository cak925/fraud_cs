import pandas as pd

def clean_addresses(row):
    if row in ['com', 'COM', 'Com', 'com ']:
        return '.com'
    elif row in ['org', 'ORG', 'Org']:
        return '.org'
    elif row in ['gov', 'us']:
        return '.gov'
    else:
        return '.other'

def extract_email_domain(df, point_predict=False):
    df['split_domain'] = df['email_domain'].apply(lambda x: x.split('.')[-1])
    df['split_domain'] = df['split_domain'].apply(clean_addresses)
    if not point_predict:
        email_dummies = pd.get_dummies(df['split_domain'], prefix = 'email')
        df = df.join(email_dummies)
    else:
       emails = ['.com', '.gov', '.org', '.other']
       df['email_.com'] = 0
       df['email_.gov'] = 0
       df['email_.org'] = 0
       df['email_.other'] = 0
       for row_idx in df.index.values:
           for email in emails:
               if df.loc[row_idx, 'split_domain'] == email:
                   col = 'email_' + email
                   df.loc[row_idx, col] = 1
    df.drop(['email_domain', 'split_domain'], axis = 1, inplace = True)
    return df