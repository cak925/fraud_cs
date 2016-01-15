import numpy as np

def get_event_ticket_info(event):
    ''' Takes an event's nested dictionary of ticket types and unpacks it.
    Meant to be used as a .apply on the 'ticket_types' column.
    Extracts and returns min/max/mean prices, total number of tickets sold, and 
    total number of tickets offered for each event. '''
    num_entries = len(event) # how many dicts to go through
    # unpack costs, quantity sold, and total quantity offered
    costs = np.array([event[i]['cost'] for i in range(num_entries)])
    quan_sold = np.array([event[i]['quantity_sold'] for i in range(num_entries)])
    quan_total = np.array([event[i]['quantity_total'] for i in range(num_entries)])
    # condense arrays by ticket price, i.e. calculate the total number of tickets
    # sold and offered at a given price (gets rid of multiple entries of 
    # tickets with the same price)
    total_tix_sold = np.array([np.sum(quan_sold[np.where(costs == cost_of_ticket)]) 
                                    for cost_of_ticket in np.unique(costs)])
    total_tix_offered = np.array([np.sum(quan_total[np.where(costs == cost_of_ticket)]) 
                                    for cost_of_ticket in np.unique(costs)])
    # could return these long arrays, along with np.unique(costs), but 
    # that will likely be too many features.
    total_revenue = np.sum(costs * quan_sold)
    try: # np.max and np.min fail for some arrays, so:
        max_cost = np.max(costs)
    except:
        max_cost = np.nan
    try:
        min_cost = np.min(costs)
    except:
        min_cost = np.nan
    return min_cost, max_cost, np.mean(costs), total_revenue, np.sum(total_tix_sold), np.sum(total_tix_offered)

def extract_info(df):
    ''' Write a new column of all info, extra info from that 
    column, then drop the original columns. '''
    df['condensed_ticket_info'] = df['ticket_types'].apply(get_event_ticket_info)
    col_to_write = ['min_price', 'max_price', 'mean_price', 'total_revenue', 'total_tix_sold', 'total_tix_offered']
    for idx, col in enumerate(col_to_write):
        df[col] = df['condensed_ticket_info'].apply(lambda x: x[idx])
    df['condensed_ticket_info'] = df['ticket_types'].apply(get_event_ticket_info)
    df.drop(['ticket_types', 'condensed_ticket_info'], axis = 1, inplace = True)
    return df