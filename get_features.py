import pandas as pd
import numpy as np
from clean_data import create_final_df

def get_features(df, point=False):

	#chosen features for model
	selected_features = ['body_length','fb_published','gts','has_analytics','has_header','has_logo',
	'name_length','num_order','num_payouts','org_facebook','org_twitter','sale_duration','sale_duration2',
	'show_map','user_age','min_price','max_price', 'mean_price', 'total_revenue', 'total_tix_sold',
	'total_tix_offered','A','C','E','G','M','N','U','num_caps_freq','email_.com','email_.gov',
	'email_.org','email_.other','delivery_method_0.0','delivery_method_1.0','delivery_method_3.0','delivery_method_nan',
	'state_GREATER_LONDON','state_FL','state_LONDON','state_GT_LON','state_DE','state_BIRMINGHAM','state_PA',
	'state_NV','state_NH','state_GA','state_ENGLAND','country_US','country_IE','country_FR','country_CA',
	'country_GB','country_AU','country_ES','country_NL','country_DE','country_VN','country_NZ','country_PK',
	'country_MA','country_A1','country_other','previous_payout']
	X = df[selected_features]
	#fill na's with -1
	X.fillna(-1, inplace = True)
	if point:
		return X.values
	else:
		#get target variable
		X = X.values
		y = df.fraud.values
		return X, y

def train_test_equal_weight(df):
    fraud_df = df.loc[df['fraud'] == 1, :]
    notfraud_df = df.loc[df['fraud'] == 0, :]

    fraud_train_df = fraud_df.sample(frac=0.8)
    fraud_test_df = fraud_df.loc[
        np.setdiff1d(fraud_df.index.values, fraud_train_df.index.values), :]

    notfraud_train_df = notfraud_df.sample(n=len(fraud_train_df))
    notfraud_test_df = notfraud_df.loc[
        np.setdiff1d(notfraud_df.index.values, notfraud_train_df.index.values), :]

    train_df = pd.concat([fraud_train_df, notfraud_train_df])
    test_df = pd.concat([fraud_test_df, notfraud_test_df])

    return train_df, test_df


if __name__ == '__main__':
    df = create_final_df()
    train_df, test_df = train_test_equal_weight(df)
    X_train, y_train = get_features(train_df)
    X_test, y_test = get_features(test_df)
    X, y = get_features(df)