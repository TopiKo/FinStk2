import pandas as pd
import os
from datetime import datetime
import numpy as np


def get_all_dfs():

    data_dir = '/home/topiko/Documents/Study/Stock/data/'

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
    dates = df_dict.keys()
    ordered_datetimes = sorted([datetime.strptime(date, '%d.%m.%Y') for date in dates])
    ordered_keys = [date.strftime("%d.%m.%Y") for date in ordered_datetimes][::-1]

    # Companies to consider
    consider_comp = sorted(list(set(df_dict[ordered_keys[0]].index.values)))
    N = len(consider_comp)

    # Columns to consider
    colls_per_company = df_dict[ordered_keys[0]].columns.values.tolist()
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
    for this_day in df.index[:n_days]:
        # Current days dataframe
        df_day = df_dict[this_day]
        # All the companies available from current day
        comps_day = df_day.index.values.tolist()
        # Set all the values from current days dataframe
        for comp_c in comps_day:
            df.loc[this_day][comp_c.strip()] = df_day.loc[comp_c]
            #df.loc[this_day][(comp_c.strip(), 'price_change')]


    comps, vals = zip(*df.columns.values.tolist())
    comps = sorted(list(set(comps)))
    vals = list(set(vals))
    vals += ['H_price_change', 'L_price_change', 'H_price_change_%', 'L_price_change_%']
    vals = sorted(vals)

    #print(vals)
    for comp in comps:
        L_vals1 = df.iloc[1:][(comp, 'Sales Lowest')]
        L_vals2 = df.iloc[:-1][(comp, 'Sales Lowest')]
        H_vals1 = df.iloc[1:][(comp, 'Sales Highest')]
        H_vals2 = df.iloc[:-1][(comp, 'Sales Highest')]
        Oe_vals1 = df.iloc[1:][(comp, 'Offer End')]
        Oe_vals2 = df.iloc[:-1][(comp, 'Offer End')]

        L_change = np.zeros(len(df))
        L_change[:-1] = L_vals2.values - L_vals1.values
        L_change_p = np.zeros(len(df))
        L_change_p[:-1] = 100*(L_vals2.values - L_vals1.values)/L_vals2.values

        H_change = np.zeros(len(df))
        H_change[:-1] = H_vals2.values - H_vals1.values
        H_change_p = np.zeros(len(df))
        H_change_p[:-1] = 100*(H_vals2.values - H_vals1.values)/H_vals2.values

        Oe_change = np.zeros(len(df))
        Oe_change[:-1] = Oe_vals2.values - Oe_vals1.values
        Oe_change_p = np.zeros(len(df))
        Oe_change_p[:-1] = 100*(Oe_vals2.values - Oe_vals1.values)/Oe_vals2.values


        df[(comp, 'L_price_change')] = L_change
        df[(comp, 'L_price_change_%')] = L_change_p
        df[(comp, 'H_price_change')] = H_change
        df[(comp, 'H_price_change_%')] = H_change_p
        df[(comp, 'Oe_price_change')] = Oe_change
        df[(comp, 'Oe_price_change_%')] = Oe_change_p

        keys = list(zip([comp]*len(vals), vals))

    return df

df = make_large_df(365*7)
df.to_pickle('combined2.pkl')
