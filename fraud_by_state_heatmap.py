from pandas import DataFrame,Series
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import random
from matplotlib.patches import Polygon, PathPatch
from matplotlib.collections import PatchCollection
from mpl_toolkits.basemap import Basemap
import json
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import csv
import sys
from clean_data import load_data
from clean_state import clean_venue_state, state_percentages
from fraud_clean import create_fraud_col

def process_data():
    ''' Load the data and apply the necessary processing to extract
    state and fraud info.'''
    df = load_data()
    df = create_fraud_col(df)
    df, state, percent, count = clean_venue_state(df, return_state_percent = True)
    all_states, perc_fraud, counts = state_percentages(df, min_fraud = 0, min_num = 0)
    states = 'AL|AK|AZ|AR|CA|CO|CT|DE|DC|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY'.split('|')
    return df, all_states, states, perc_fraud, counts

def get_state_percentages(all_states, states, perc_fraud, counts):
    ''' Subset original list of all states (including international data mess)
    to just US, along with the corresponding data. '''
    state_subset_idx = np.where(np.in1d(all_states, states))
    us_state_perc = perc_fraud[state_subset_idx[0]]
    us_state_counts = counts[state_subset_idx[0]]
    us_state_names = all_states[state_subset_idx[0]]
    return us_state_names, us_state_perc, us_state_counts

def add_subplot_axes(ax,rect):
    ''' Code to add subplots for Alaska and Hawaii. '''
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    subax = fig.add_axes([x,y,width,height],frameon=False)
    return subax

def load_state_codes():
    ''' Necessary dictionaries to plot shapefiles and convert between formats. ''' 
    state_codes = { '01': 'Alabama',
                    '02': 'Alaska',
                    '15': 'Hawaii',
                    '04': 'Arizona',
                    '05': 'Arkansas',
                    '06': 'California',
                    '08': 'Colorado',
                    '09': 'Connecticut',
                    '10': 'Delaware',
                    #   '11': 'District of Columbia',
                    '12': 'Florida',
                    '13': 'Georgia',
                    '16': 'Idaho',
                    '17': 'Illinois',
                    '18': 'Indiana',
                    '19': 'Iowa',
                    '20': 'Kansas',
                    '21': 'Kentucky',
                    '22': 'Louisiana',
                    '23': 'Maine',
                    '24': 'Maryland',
                    '25': 'Massachusetts',
                    '26': 'Michigan',
                    '27': 'Minnesota',
                    '28': 'Mississippi',
                    '29': 'Missouri',
                    '30': 'Montana',
                    '31': 'Nebraska',
                    '32': 'Nevada',
                    '33': 'New Hampshire',
                    '34': 'New Jersey',
                    '35': 'New Mexico',
                    '36': 'New York',
                    '37': 'North Carolina',
                    '38': 'North Dakota',
                    '39': 'Ohio',
                    '40': 'Oklahoma',
                    '41': 'Oregon',
                    '42': 'Pennsylvania',
                    '44': 'Rhode Island',
                    '45': 'South Carolina',
                    '46': 'South Dakota',
                    '47': 'Tennessee',
                    '48': 'Texas',
                    '49': 'Utah',
                    '50': 'Vermont',
                    '51': 'Virginia',
                    '53': 'Washington',
                    '54': 'West Virginia',
                    '55': 'Wisconsin',
                    '56': 'Wyoming'}
    state_conversion = {
                    'Alabama': 'AL',
                    'Alaska': 'AK',
                    'Arizona': 'AZ',
                    'Arkansas': 'AR',
                    'California': 'CA',
                    'Colorado': 'CO',
                    'Connecticut': 'CT',
                    'Delaware': 'DE',
                    'District of Columbia': 'DC',
                    'Florida': 'FL',
                    'Georgia': 'GA',
                    'Hawaii': 'HI',
                    'Idaho': 'ID',
                    'Illinois': 'IL',
                    'Indiana': 'IN',
                    'Iowa': 'IA',
                    'Kansas': 'KS',
                    'Kentucky': 'KY',
                    'Louisiana': 'LA',
                    'Maine': 'ME',
                    'Maryland': 'MD',
                    'Massachusetts': 'MA',
                    'Michigan': 'MI',
                    'Minnesota': 'MN',
                    'Mississippi': 'MS',
                    'Missouri': 'MO',
                    'Montana': 'MT',
                    'National': 'NA',
                    'Nebraska': 'NE',
                    'Nevada': 'NV',
                    'New Hampshire': 'NH',
                    'New Jersey': 'NJ',
                    'New Mexico': 'NM',
                    'New York': 'NY',
                    'North Carolina': 'NC',
                    'North Dakota': 'ND',
                    'Ohio': 'OH',
                    'Oklahoma': 'OK',
                    'Oregon': 'OR',
                    'Pennsylvania': 'PA',
                    'Puerto Rico': 'PR',
                    'Rhode Island': 'RI',
                    'South Carolina': 'SC',
                    'South Dakota': 'SD',
                    'Tennessee': 'TN',
                    'Texas': 'TX',
                    'Utah': 'UT',
                    'Vermont': 'VT',
                    'Virginia': 'VA',
                    'Washington': 'WA',
                    'West Virginia': 'WV',
                    'Wisconsin': 'WI',
                    'Wyoming': 'WY'}   
    state_populations = {
                    'Alabama': 4849377,
                    'Alaska': 737732,
                    'Arizona': 6731484,
                    'Arkansas': 2994079,
                    'California': 38802500,
                    'Colorado': 5355856,
                    'Connecticut': 3596677,
                    'Delaware': 935614,
                    'Florida': 19893297,
                    'Georgia': 10097343,
                    'Hawaii': 1419561,
                    'Idaho': 1634464,
                    'Illinois': 12880580,
                    'Indiana': 6596855,
                    'Iowa': 3107126,
                    'Kansas': 2904021,
                    'Kentucky': 4413457,
                    'Louisiana': 4649676,
                    'Maine': 1330089,
                    'Maryland': 5976407,
                    'Massachusetts': 6745408,
                    'Michigan': 9909877,
                    'Minnesota': 5457173,
                    'Mississippi': 2984926,
                    'Missouri': 6063589,
                    'Montana': 1023579,
                    'Nebraska': 1881503,
                    'Nevada': 2839099,
                    'New Hampshire': 1326813,
                    'New Jersey': 8938175,
                    'New Mexico': 2085572,
                    'New York': 19746227,
                    'North Carolina': 9943964,
                    'North Dakota': 739482,
                    'Ohio': 11594163,
                    'Oklahoma': 3878051,
                    'Oregon': 3970239,
                    'Pennsylvania': 12787209,
                    'Rhode Island': 1055173,
                    'South Carolina': 4832482,
                    'South Dakota': 853175,
                    'Tennessee': 6549352,
                    'Texas': 26956958,
                    'Utah': 2942902,
                    'Vermont': 626011,
                    'Virginia': 8326289,
                    'Washington': 7061530,
                    'West Virginia': 1850326,
                    'Wisconsin': 5757564,
                    'Wyoming': 584153}
    return state_codes, state_conversion, state_populations

def plot_USA_map(state_codes, state_conversion, state_subset, percentage_fraud, counts_fraud, state_populations, typ = 'Rate'):
    ''' Assign colors to each state shapefile based on some metric,
    normalized to whatever the max of that metric is. Add plot insets
    for Hawaii and Alaska.'''
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111)
    rect = [0.08,0.05,0.35,0.35]
    axAlaska = add_subplot_axes(ax,rect)
    rect = [0.3,0.02,0.2,0.2]
    axHawaii = add_subplot_axes(ax,rect)

    fig.suptitle('Fraud {} By State'.format(typ), fontsize=20)

    mNormal = Basemap(width=5000000,height=3500000,
                resolution='l',projection='aea',\
                ax=ax, \
                lon_0=-96,lat_0=38)

    mAlaska = Basemap(width=5000000,height=3500000,
                resolution='l',projection='aea',\
                ax=axAlaska, \
                lon_0=-155,lat_0=65)

    mHawaii = Basemap(width=1000000,height=700000,
                resolution='l',projection='aea',\
                ax=axHawaii, \
                lon_0=-157,lat_0=20)

    # define a colormap
    num_colors = 101
    cm = plt.get_cmap('summer')
    colors = [cm(1.*i/num_colors) for i in range(num_colors)]
    if typ == 'Rate':
        max_fraud = percentage_fraud.max()
    else:
        max_fraud = counts_fraud.max()

    # read each states shapefile
    for key in state_codes.keys():
        state_name = state_codes[key]
        print state_name
        if (state_codes[key] == "Alaska"):
            mAlaska.readshapefile('shapefiles_take2/tl_2013_{0}_puma10/tl_2013_{0}_puma10'.format(key),name='state', drawbounds=True)
            m = mAlaska
        elif (state_codes[key] == "Hawaii"):
            mHawaii.readshapefile('shapefiles_take2/tl_2013_{0}_puma10/tl_2013_{0}_puma10'.format(key),name='state', drawbounds=True)
            m = mHawaii
        else:
            mNormal.readshapefile('shapefiles_take2/tl_2013_{0}_puma10/tl_2013_{0}_puma10'.format(key),name='state', default_encoding='latin-1', drawbounds=True)
            m = mNormal
        state_key = state_conversion[state_name]
        if typ == 'Rate':
            state_fraud_rate = percentage_fraud[np.where(state_subset == state_key)]
            color_idx = int(100 * state_fraud_rate[0] / max_fraud)
        elif typ == 'Rate Normalized by Population':
            state_pop = state_populations[state_name]
            state_fraud_normalized = counts_fraud[np.where(state_subset == state_key)][0] / float(state_pop)
            print state_fraud_normalized
            color_idx = int(100 * state_fraud_normalized / 3.85447403626e-05)
        else:
            state_fraud_count = counts_fraud[np.where(state_subset == state_key)]
            color_idx = int(100 * state_fraud_count[0] / max_fraud)
        color = colors[color_idx]
        for info, shape in zip(m.state_info, m.state):
            patches = [Polygon(np.array(shape), True)]
            pc = PatchCollection(patches, edgecolor='k', linewidths=1., zorder=2)
            pc.set_color(color)
            if (state_codes[key] == "Alaska"):
                axAlaska.add_collection(pc)
            elif (state_codes[key] == "Hawaii"):
                axHawaii.add_collection(pc)
            else:
                ax.add_collection(pc)

    # add colorbar legend
    cmap = mpl.colors.ListedColormap(colors)
    # define the bins
    bounds = np.linspace(0, max_fraud, 20) 
    # create a second axes for the colorbar
    ax2 = fig.add_axes([0.82, 0.1, 0.03, 0.8])
    cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, ticks=bounds, boundaries=bounds, format='%1i')
    legend_str = [str(float(i)).split('.')[0] for i in bounds]
    # cb.ax.set_yticks([legend_str])
    cb.ax.set_yticklabels(legend_str)# vertically oriented colorbar
    plt.savefig('{}_fraud_by_state.png'.format(typ), dpi = 200)

if __name__ == '__main__':
    df, all_states, states, perc_fraud, counts = process_data()
    state_subset, us_state_perc, us_state_counts = get_state_percentages(all_states, states, perc_fraud, counts)
    state_codes, state_conversion, state_populations= load_state_codes()
    # plot_USA_map(state_codes, state_conversion, state_subset, us_state_perc, us_state_counts, state_populations,  typ = 'Rate Normalized by Population')
    plot_USA_map(state_codes, state_conversion, state_subset, us_state_perc, us_state_counts, state_populations, typ = 'Count')
Status API Training Shop Blog About Pricing
© 2016 GitHub, Inc. Terms Privacy Security Contact Help