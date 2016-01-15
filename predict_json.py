import json
import pandas as pd
import numpy as np
import cPickle as pickle

from date_cleaner import date_clean
from fraud_clean import create_fraud_col
from split_ticket_types import extract_info
from text_work import text_work
from text_work2 import get_num_caps
from currDum_saleDur_evLen import currency_and_duration
from extract_email_domain import extract_email_domain
from delivery_dummies import get_delivery_dummies
from clean_state import clean_venue_state
from clean_country import make_country_dummies_point
from previous_payouts import previous_payout
from get_features import get_features


def load_model(path):
    # Load in pickled model
    with open(path) as f:
        model = pickle.load(f)
    return model


def process_json(datapoint, model):
    '''
    INPUT: JSON DICT FROM JSON REQUEST IN FLASK FILE
    OUTPUT: JSON DICT AFTER PROCESSING IT
    '''
    raw_json = json.loads(datapoint)
    raw_array = np.array(raw_json.values())
    raw_array = raw_array.reshape(1, -1)

    df = pd.DataFrame(data=raw_array, columns=raw_json.keys())
    df = process_point(df)
    prediction, prob = predict(df, model)
    return prediction, prob


def process_point(df):
    df = date_clean(df)
    df = extract_info(df)
    df = currency_and_duration(df, point_predict=True)
    df = get_num_caps(df)
    df = extract_email_domain(df, point_predict=True)
    df = get_delivery_dummies_point(df)
    df = clean_venue_state(df, point_predict=True)
    df = make_country_dummies_point(df)
    # Creates a boolean columns whether there is a previous payout or not as well as dropping the previous_payouts column
    df = previous_payout(df)
    return df



def predict(df, model):
    '''
    INPUT: Processed datapoint / dataframe
    OUTPUT: Array of Predictions, Array of Probabilities
    '''
    X = get_features(df, point=True)
    predictions = model.predict(X)
    probs = model.predict_proba(X)
    return predictions, probs