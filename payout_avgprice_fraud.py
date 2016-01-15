import pandas as pd
import numpy as np
from date_cleaner import date_clean
from fraud_clean import create_fraud_col
from split_ticket_types import extract_info
from text_work import text_work
from currDum_saleDur_evLen import currency_and_duration
from extract_email_domain import extract_email_domain
from delivery_dummies import get_delivery_dummies
from clean_state import clean_venue_state
from clean_country import make_country_dummies
from previous_payouts import previous_payout
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns


def load_data():
    df = pd.read_json("../fraud-detection-case-study/data/train_new.json")
    return df


def process_df(df):
    df = date_clean(df)
    df = create_fraud_col(df)
    df = extract_info(df)
    df = currency_and_duration(df)
    df = text_work(df)
    df = extract_email_domain(df)
    df = get_delivery_dummies(df)
    df = clean_venue_state(df)
    df = make_country_dummies(df)
    return df


def num_previous_payouts(df):
    def num_payouts(row):
        return len(row)
    df['num_previous_payouts'] = df['previous_payouts'].apply(num_payouts)
    return df


def plot_payout_vs_price(df, max_payout_num=200):
    # Cut the dataframe down to just max_payout_num or less previous payouts
    sub_df = df.loc[df['num_previous_payouts'] <= max_payout_num, :]

    unique_payouts = np.sort(sub_df['num_previous_payouts'].unique())
    avg_mean_price = np.array(
        [df.loc[df['num_previous_payouts'] == payout, 'mean_price'].mean() for payout in unique_payouts])
    percent_fraud = np.array(
        [df.loc[df['num_previous_payouts'] == payout, 'fraud'].mean() for payout in unique_payouts])

    plt.scatter(unique_payouts, avg_mean_price,
                s=percent_fraud * 1200, c=percent_fraud, cmap=cm.Reds)
    plt.title('Number of Payouts Vs. Avg. Mean Price')
    plt.xlabel('Number of Payouts Up To {} Payouts'.format(max_payout_num))
    plt.ylabel('Avg. Mean Price (Size Indicates % Fraud)')
    plt.xlim(-3, max_payout_num)
    plt.ylim(-5, 250)
    plt.colorbar(label='% of Fraud')
    plt.savefig('./payout_avgprice_fraud.jpg', dpi=200)
    plt.clf()


if __name__ == '__main__':
    df = load_data()
    df = process_df(df)

    df = num_previous_payouts(df)

    plot_payout_vs_price(df, 35)