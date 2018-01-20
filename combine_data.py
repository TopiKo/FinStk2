import pandas as pd
import os
from datetime import datetime
import numpy as np


def get_all_dfs():

    data_dir = '/home/topiko/Workspace/FinStk2/data/'

    all_companies = set()
    df_dict = {}

    for f_name in os.listdir(data_dir):
        if f_name.endswith('.pkl'):
            df = pd.read_pickle(data_dir + f_name)

            date = f_name[12:-4]
            df_dict[date] = df
            a =  set(df.index.values)
            if len(df.index) != len(a):
                print('not unique!')
                raise
            all = set(all_companies | a)

    return df_dict


def make_large_df(n_days = 0):

    # Get set of df in a dict where index is date and value datframe
    df_dict = get_all_dfs()
    if n_days == 0: n_days = len(df_dict)

    # Order the dates
    ordered_datetimes = sorted([datetime.strptime(date, '%d.%m.%Y') for date in df_dict.keys()])
    ordered_keys = [date.strftime("%d.%m.%Y") for date in ordered_datetimes][::-1]

    # Companies to consider
    consider_comp = sorted(list(set(df_dict[ordered_keys[1]].index.values)))
    N = len(consider_comp)

    # Columns to consider
    #colls_per_company = df_dict[ordered_keys[0]].columns.values.tolist()
    #print(colls_per_company) ['Offer End', 'Offer Buy', 'Offer Sell', 'Sales Lowest', 'Sales Highest', 'Change M. Eur']
    colls_per_company = ['offer_end', 'offer_buy', 'offer_sell', 'sales_low', 'sales_high', 'change_Me']
    nc = len(colls_per_company)

    # Make array with each company name repeated 6 times (once for for each column).
    comps = np.reshape(np.array([np.repeat(comp.strip(), nc) for comp in consider_comp]), N*nc)

    # Dual column indices:
    index_arrs = [comps, [val for val in colls_per_company]*N]
    tuples = list(zip(*index_arrs))
    index = pd.MultiIndex.from_tuples(tuples, names=['company', 'values'])

    # Init nan array for the large df
    init_nans = np.empty((n_days, len(comps)))
    init_nans[:,:] = np.nan

    # Make df with columns having multi index (company, data) and index, date.
    df = pd.DataFrame(init_nans, index=ordered_keys[:n_days], columns=index)
    df.index.name = 'date'

    # Set values for each date into the large df
    for this_day in df.index.values:
        # All the companies available from current day
        df_day = df_dict[this_day]
        comps_day = df_day.index.values.tolist()
        print(this_day, len(comps_day))
        for comp_c in comps_day:
            # Set all the values from current days dataframe
            df.loc[this_day][comp_c.strip()] = df_day.loc[comp_c]


    comps, vals = zip(*df.columns.values.tolist())
    print(set(comps))
    for comp in set(comps):
        slow1, slow0 = df.iloc[1:][(comp, 'sales_low')].values, df.iloc[:-1][(comp, 'sales_low')].values
        shigh1, shigh0 = df.iloc[1:][(comp, 'sales_high')].values, df.iloc[:-1][(comp, 'sales_high')].values
        oend1, oend0 = df.iloc[1:][(comp, 'offer_end')].values, df.iloc[:-1][(comp, 'offer_end')].values

        c_slow = slow0 - slow1
        c_slow_p = c_slow/slow1

        c_shigh = shigh0 - shigh1
        c_shigh_p = c_shigh/shigh1

        c_oend = oend0 - oend1
        c_oend_p = c_oend/oend1

        df[(comp, 'c_slow')] = np.append(c_slow, np.nan)
        df[(comp, 'c_slow_%')] = np.append(c_slow_p, np.nan)
        df[(comp, 'c_shigh')] = np.append(c_shigh, np.nan)
        df[(comp, 'c_shigh_%')] = np.append(c_shigh_p, np.nan)
        df[(comp, 'c_oend')] = np.append(c_oend, np.nan)
        df[(comp, 'c_oend_%')] = np.append(c_oend_p, np.nan)

    # lose the last row of the df, as the change is not known there...
    df = df[:-1]

    return df

df = make_large_df(int(365*3.02))

df.to_pickle('combined.pkl')
