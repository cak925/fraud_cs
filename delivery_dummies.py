import pandas as pd


def get_delivery_dummies(df):
    dummies = pd.get_dummies(
        df['delivery_method'], prefix='delivery_method', dummy_na=True)
    df.drop('delivery_method', axis=1, inplace=True)
    return df.join(dummies)

delivery_method_0.0','delivery_method_1.0','delivery_method_3.0','delivery_method_nan

def get_delivery_dummies_point(df):
    methods = [0.0, 1.0, 3.0]
    for method in methods:
        col = 'delivery_method_' + str(method)
        df[col] = 0
    df['delivery_method_nan'] = 0

    for row_idx in df.index.values:
        set_val = False
        for method in methods:
            if df.loc[row_idx, 'delivery_method'] == method:
                col = 'delivery_method_' + method
                df.loc[row_idx, col] = 1
        if not set_val:
            df.loc[row_idx, 'delivery_method_nan'] = 1
    df.drop('delivery_method', axis=1, inplace=True)
    return df