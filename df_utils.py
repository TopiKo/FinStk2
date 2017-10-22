import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, confusion_matrix, roc_auc_score, precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.decomposition import PCA
import itertools

def get_X_y(df, predComp, nfut, nhist, totHist):

    comp_cols = df[predComp].columns
    nfut -= 1

    # Number of to become features
    nfeat = (nfut + nhist + 1)*len(comp_cols)

    data_arr = np.zeros((totHist, nfeat))
    indices = np.empty(totHist, dtype = 'object')

    for k, i in enumerate(range(nfut, totHist + nfut)):
        data_arr[k,:] = np.reshape(df.iloc[i-nfut:i+nhist+1][predComp].values,
                          nfeat, order = 'F')
        indices[k] = df.index.values[i]

    # Full dataframe with all features also the future
    X = pd.DataFrame(data = data_arr, index = indices) #.from_dict(data_dict, orient = 'index')
    X.columns = ['{:s}_{:03d}'.format(col, -i) for col in comp_cols for i in range(-nfut, nhist + 1)]

    # Find the future vals mask
    X_cols_mask = [int(col[-3:]) < 0 for col in X.columns]
    y_cols_mask = np.logical_not(X_cols_mask)

    # Pick the future and add previous day offer end
    y = X[X.columns[y_cols_mask]].copy()
    y.loc[:,'offer_end_prev'] = X['offer_end_-01'].values
    y.loc[:,'sales_low_prev'] = X['sales_low_-01'].values
    y.loc[:,'sales_high_prev'] = X['sales_high_-01'].values

    # Fill still existing possible nans
    X.fillna(method='bfill', inplace = True)
    y.fillna(method='bfill', inplace = True)
    X.fillna(method='ffill', inplace = True)
    y.fillna(method='ffill', inplace = True)

    y, ysim = refine_y(y, nfut)

    return X[X.columns[X_cols_mask]], y, ysim

def refine_y(y,nfut):
    '''
    Create the values you want to predict
    '''

    y.loc[:,'offer_end_change'] = (y['offer_end_{:03d}'.format(nfut)] \
                                - y['offer_end_prev'])/y['offer_end_prev']

    y.loc[:,'sale_low_to_high_change'] = (y['sales_low_{:03d}'.format(nfut)] \
                                - y['sales_high_000'])/y['sales_high_000']

    y.loc[:,'sale_low_change'] = (y['sales_low_{:03d}'.format(nfut)] \
                                - y['sales_low_prev'])/y['sales_low_prev']

    y.loc[:,'sale_high_change'] = (y['sales_high_{:03d}'.format(nfut)] \
                                - y['sales_high_prev'])/y['sales_high_prev']

    # , 'c_oend_%_000', 'c_oend_%_001'
    return y[['offer_end_change','sale_low_to_high_change',
              'sale_low_change', 'sale_high_change']], \
            y[['offer_end_change', 'sale_low_to_high_change', 'sale_low_change',
            'offer_end_prev', 'sales_low_000', 'sales_high_000', 'offer_end_{:03d}'.format(nfut)]]


def get_companies_list(nan_streak_threshold = 10):

    df = pd.read_pickle('combined.pkl')

    # return companies that do not have too many nans...
    l1, l2 = zip(*df.columns.values)
    companies = list(set(l1))
    values = list(set(l2))
    exclude = []
    for comp in companies:
        for i in range(len(values)):
            nans = np.isnan(df[comp].values[:,i]).astype(int)
            nan_streak = [sum( 1 for _ in group ) for key, group in itertools.groupby(nans) if key]
            if len(nan_streak) != 0:
                if nan_streak_threshold < max(nan_streak):
                    exclude.append(comp)

    companies = sorted([company for company in companies if company not in exclude])


    return df[companies], companies

#df, comps = get_companies_list(2)
#get_X_y(df, 'Alma Media', 2, 10, 2*365)
