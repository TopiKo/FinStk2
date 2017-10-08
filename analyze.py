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

def make_Xy(df, consider, predict, timeDelta = 5, futureTime = 1, timeHistory = 365):

    #aheadTime #of days before we make prediction
    #timeDelta #of previous days to consider when making prediction

    df.reset_index(inplace = True)

    # Dual column indices:
    # Make new indexing as the df is going to be formatted heavily.
    # Rows timeDelta days will be transformed into columns with data and day labels.
    comps = np.reshape(np.array([np.repeat(comp[0].strip(), timeDelta) for comp in consider]), len(consider)*timeDelta)
    arr = np.reshape(np.array([['{:s}_{:d}'.format(val[1],-(i+1)) for i in range(timeDelta)] for val in consider]), timeDelta*len(consider))

    col_tuples = list(zip(*[comps, arr]))
    index = pd.MultiIndex.from_tuples(col_tuples, names=['company', 'values'])
    df_X = pd.DataFrame(columns = index)
    #df_y = pd.Series(name = predict[0])
    df_y = pd.DataFrame(columns = range(futureTime))
    n = 0

    for i in range(futureTime + 1, timeHistory + futureTime + 1):
        # pick data from days that are BEFORE the now considered day i. The dates are arranged
        # descendind order.
        df_l = df.iloc[i : i + timeDelta][consider]
        # Reshape the matrix into one long vector
        data = np.reshape(df_l.values, timeDelta*len(consider), order = 'F')
        df_X.loc[n] = data
        #df_y.loc[i] = df.iloc[i][predict].values.tolist()[0]

        # For the prediction pick values TODAY and some days (futureTime) ahead.
        df_y.loc[n] =[1 + .01*df.iloc[i - 1 - add][predict].values[0] for add in range(futureTime)]
        n += 1
        #print(data, (df_y.loc[i].values - 1)*100)
    #df_X.drop(predict[0][0], axis = 1, inplace = True)

    return df_X, df_y


def get_Xy(init_features, predict, pick_comps = [], timeDelta = 5, futureTime = 3, timeHistory = 365):

    df = pd.read_pickle('combined.pkl')
    l1, l2 = zip(*df.columns.values)
    companies = list(set(l1))
    values = list(set(l2))

    l1, l2 = zip(*df.columns.values.tolist())

    # Predict with these:
    # values_pick = ['Oe_price_change_%', 'Change M. Eur'] #'L_price_change_%' 'Change M. Eur'

    excl_comp_list = ['Ahola Transport A', 'Aktia Pankki R', 'Asiakastieto Group', 'Consti Yhtiöt',
                      'DNA', 'Detection Technology', 'Digitalist Group', 'Dovre Group',
                      'Elite Varainhoito', 'Evli Pankki', 'Elecster A','Ericsson B',
                      'FIT Biotech', 'Fondia', 'Heeros', 'Ilkka-Yhtymä I', 'Ilkka-Yhtymä II',
                      'Kamux Oyj', 'Kotipizza Group', 'Kesla A',
                      'Lehto Group', 'Nexstim', 'Next Games', 'Nixu', 'Nurminen Logistics', 'Pihlajalinna',
                      'Piippo', 'Pohjois-Karjalan Kirjapaino', 'PKC Group', 'Privanet Group',
                      'Qt Group', 'Remedy Entertainment',
                      'Robit', 'Savo-Solar', 'Silmäasema Oyj', 'Soprano', 'Sievi Capital',
                      'Suomen Hoivatilat', "Trainers' House", 'Tulikivi A','Tecnotree',
                      'Talenom', 'Talvivaara', 'Tokmanni Group', 'United Bankers',
                      'Vincit Group', 'Yleiselektroniikka E', 'Wulff-Yhtiöt', 'Zeeland Family',
                      'Ålandsbanken A']

    if len(pick_comps) != 0:
        rem_comps2 = [comp for comp in companies if comp not in pick_comps]
        excl_comp_list += rem_comps2

    for comp in list(set(excl_comp_list)):
        companies.remove(comp)



    companies_pick = np.repeat(np.array(sorted(companies)), len(init_features))
    #predict = [(comp_predict, val_predict)]

    consider = list(zip(*[companies_pick, init_features*len(companies)]))
    dfX, dfy = make_Xy(df, consider, predict, timeDelta, futureTime, timeHistory)

    return dfX, dfy, df

def plot_stock(df, comp, features  = ['Offer Sell']):

    #x = y.index.values.tolist()
    x = -df.index.values
    plt.figure(figsize = (20, 7))
    [y0, y1] = [df[(comp, features[i])].values for i in range(len(features))]
    ypred_mask = df[(comp, 'prediction')].values
    ytruth_mask = df[(comp, 'ground_truth')].values

    correct_pred_mask = ypred_mask & ytruth_mask
    incorrect_pred_mask = ypred_mask & np.logical_not(ytruth_mask)

    X_pred_cor = x[correct_pred_mask.tolist()]
    Y_pred_cor = y0[correct_pred_mask.tolist()]
    X_pred_incor = x[incorrect_pred_mask.tolist()]
    Y_pred_incor = y0[incorrect_pred_mask.tolist()]


    plt.plot(x,y0, label = features[0])
    plt.scatter(X_pred_cor, Y_pred_cor)
    plt.scatter(X_pred_incor, Y_pred_incor, s = 100, c = 'red')
    #plt.scatter(x[ytruth_mask.tolist()], y0[ytruth_mask.tolist()], c = 'red')

    for xc in x:
        plt.axvline(x=xc, alpha = .2)
    plt.legend(loc = 2, frameon = False)


    ax2 = plt.gca().twinx()
    ax2.plot(x,y1, c = 'red', label = features[1])

    plt.xlabel('days')
    plt.title(comp)
    plt.legend(loc = 1, frameon = False)
    plt.show()

def fit_Learner(comp, pick_comps, threshold = .5, plot_stock_b = False):

    #comp = 'Affecto'

    nfut = 2
    ndays = 10
    nhist = 365*2

    # 'L_price_change', 'L_price_change_%', 'H_price_change',
    # 'Oe_price_change', 'Offer Sell', 'H_price_change_%',
    # 'Change M. Eur', 'Sales Highest', 'Oe_price_change_%',
    # 'Offer Buy', 'Offer End', 'Sales Lowest'
    init_features = ['Oe_price_change_%', 'L_price_change_%', 'H_price_change_%', 'Change M. Eur'] #'L_price_change_%' 'Change M. Eur'
    predict = [(comp, 'L_price_change_%')]


    X, y, df = get_Xy(init_features, predict, pick_comps, ndays, nfut, nhist)
    col_names = X[comp].columns.values.tolist()
    #y['1x2'] = y[0].multiply(y[1], axis="index")

    y = (y.prod(axis = 1) - 1)*100

    #y = y.max(axis = 1)

    #

    bad = []
    for col in X.columns.values.tolist():
        if 10 < sum(X[col].isnull()):
            if col[0] not in bad: bad.append(col[0])

    if 5 < sum(y.isnull()):
        print('YYYYYYYYYYYYYYYYYYYYY')

    X.fillna(method='bfill', inplace = True)
    y.fillna(method='bfill', inplace = True)
    X.fillna(method='ffill', inplace = True)
    y.fillna(method='ffill', inplace = True)


    '''
    poly = PolynomialFeatures(degree=2, interaction_only=True)
    X = poly.fit_transform(X)
    rf = RandomForestClassifier(n_estimators = 750, max_depth = 12, max_features = 7)
    '''

    pipe = make_pipeline(PolynomialFeatures(), PCA(), RandomForestClassifier())
    #print(X)

    y =  y > threshold
    X_train = X[int(nhist*.25):]
    y_train = y[int(nhist*.25):]
    X_test = X[:int(nhist*.25)]
    y_test = y[:int(nhist*.25)]

    #l1, l2 = zip(*X_train.columns.values.tolist())
    #print(list(l1))

    param_grid = [{'polynomialfeatures__degree': [1],
                  'pca__n_components': [40],
                  'randomforestclassifier__max_features': [7],
                  'randomforestclassifier__max_depth': [5],
                  'randomforestclassifier__n_estimators': [50]}]
                  #{'polynomialfeatures__degree': [2],
                  # 'pca__n_components': [25, 40, 80],
                  # 'randomforestclassifier__max_features': [3, 5, 7],
                  # 'randomforestclassifier__max_depth': [3, 5],
                  # 'randomforestclassifier__n_estimators': [10, 50, 100]} ]

    grid_rf = GridSearchCV(pipe, param_grid, scoring='roc_auc', verbose = 0, cv = 2, n_jobs = 2)
    #grid_rf = rf
    grid_rf.fit(X_train, y_train)

    y_pred = grid_rf.predict(X_test)
    y_pred_p = grid_rf.predict_proba(X_test)[:,1]

    y_pred_train = grid_rf.predict(X_train)
    y_pred_p_train = grid_rf.predict_proba(X_train)[:,1]

    tmp_y_train = pd.Series(y_pred_train, index = X_train.index)
    tmp_y_test = pd.Series(y_pred, index = X_test.index)

    df = df[nfut+1:]
    df.reset_index(inplace = True)
    df[(comp, 'prediction')] = pd.concat([tmp_y_test, tmp_y_train])
    df[(comp, 'ground_truth')] = y


    if plot_stock_b: plot_stock(df.iloc[:nhist], comp, ['Sales Lowest', 'L_price_change_%'])


    fpr, tpr, _ = roc_curve(y_test, y_pred_p)
    roc_auc_s = roc_auc_score(y_test, y_pred_p)
    precision_s = precision_score(y_test, y_pred)

    roc_auc_s_train = roc_auc_score(y_train, y_pred_p_train)

    print(grid_rf.best_params_)
    print('{:s}: precision = {:.2f}, roc_auc_score = {:.2f}, roc_auc_score_train = {:.2f}'.format(comp,
     precision_s, roc_auc_s, roc_auc_s_train))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print()



    #plt.show()
    return roc_auc_s



def fit_predict(X_train, X_test, y_train, y_test, comp, pick_comps, threshold = .5, plot_stock_b = False):

    #comp = 'Affecto'


    pipe = make_pipeline(PolynomialFeatures(), PCA(), RandomForestClassifier())


    #l1, l2 = zip(*X_train.columns.values.tolist())
    #print(list(l1))

    param_grid = [{'polynomialfeatures__degree': [1],
                  'pca__n_components': [40],
                  'randomforestclassifier__max_features': [7],
                  'randomforestclassifier__max_depth': [5],
                  'randomforestclassifier__n_estimators': [50]}]
                  #{'polynomialfeatures__degree': [2],
                  # 'pca__n_components': [25, 40, 80],
                  # 'randomforestclassifier__max_features': [3, 5, 7],
                  # 'randomforestclassifier__max_depth': [3, 5],
                  # 'randomforestclassifier__n_estimators': [10, 50, 100]} ]

    grid_rf = GridSearchCV(pipe, param_grid, scoring='roc_auc', verbose = 0, cv = 2, n_jobs = 2)
    #grid_rf = rf
    grid_rf.fit(X_train, y_train)

    y_pred = grid_rf.predict(X_test)
    y_pred_p = grid_rf.predict_proba(X_test)[:,1]

    y_pred_train = grid_rf.predict(X_train)
    y_pred_p_train = grid_rf.predict_proba(X_train)[:,1]

    tmp_y_train = pd.Series(y_pred_train, index = X_train.index)
    tmp_y_test = pd.Series(y_pred, index = X_test.index)

    '''
    df = df[nfut+1:]
    df.reset_index(inplace = True)
    df[(comp, 'prediction')] = pd.concat([tmp_y_test, tmp_y_train])
    df[(comp, 'ground_truth')] = y


    if plot_stock_b: plot_stock(df.iloc[:nhist], comp, ['Sales Lowest', 'L_price_change_%'])
    '''


    fpr, tpr, _ = roc_curve(y_test, y_pred_p)
    roc_auc_s = roc_auc_score(y_test, y_pred_p)
    precision_s = precision_score(y_test, y_pred)

    roc_auc_s_train = roc_auc_score(y_train, y_pred_p_train)

    print(grid_rf.best_params_)
    print('{:s}: precision = {:.2f}, roc_auc_score = {:.2f}, roc_auc_score_train = {:.2f}'.format(comp,
     precision_s, roc_auc_s, roc_auc_s_train))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print()



    #plt.show()
    return y_pred_p, roc_auc_s


def get_train_test(comp, pick_comps):

    nfut = 2
    ndays = 10
    nhist = 365*2

    # 'L_price_change', 'L_price_change_%', 'H_price_change',
    # 'Oe_price_change', 'Offer Sell', 'H_price_change_%',
    # 'Change M. Eur', 'Sales Highest', 'Oe_price_change_%',
    # 'Offer Buy', 'Offer End', 'Sales Lowest'
    init_features = ['Sales Lowest', 'Sales Highest', 'Oe_price_change_%', 'L_price_change_%', 'H_price_change_%', 'Change M. Eur'] #'L_price_change_%' 'Change M. Eur'
    predict = [(comp, 'L_price_change_%')]

    X, y, df = get_Xy(init_features, predict, pick_comps, ndays, nfut, nhist)
    y = (y.prod(axis = 1) - 1)*100


    bad = []
    for col in X.columns.values.tolist():
        if 10 < sum(X[col].isnull()):
            if col[0] not in bad: bad.append(col[0])

    if 5 < sum(y.isnull()):
        print('YYYYYYYYYYYYYYYYYYYYY')

    X.fillna(method='bfill', inplace = True)
    y.fillna(method='bfill', inplace = True)
    X.fillna(method='ffill', inplace = True)
    y.fillna(method='ffill', inplace = True)

    y =  y > threshold
    X_train = X[int(nhist*.25):]
    y_train = y[int(nhist*.25):]
    X_test = X[:int(nhist*.25)]
    y_test = y[:int(nhist*.25)]

    return X_train, X_test, y_train, y_test, df

def invest_to(inv_list):

    max_pred = 0
    for val0,val1 in inv_list:
        if max_pred < val1:
            max_pred = val1
            comp = val0
    return comp

def test_simulate(comp_dict, fund = 10000):

    col_list = ['company', 'buy/sell', 'n', 'price', 'id']
    shop_df = pd.DataFrame(columns = col_list)
    comps = list(comp_dict.keys())
    free_money = fund
    bought_days = []
    shop_dict = {}
    id = 0
    for day in range(len(comp_dict[comps[0]])):
        invest = []

        for comp in comps:
            if comp_dict[comp].iloc[day]['prediction'] > 0.5:
                invest.append((comp, comp_dict[comp].iloc[day]['prediction']))

        day_prices = comp_dict[comp].iloc[day][['Sales Lowest', 'Sales Highest']].values

        if 0 < len(invest):
            comp = invest_to(invest)
            n_buy = int(free_money/day_prices[1]/2)
            if n_buy > 0:
                free_money -= n_buy*day_prices[1]
                shop_dict['buy_%i' %day] = [comp, -n_buy*day_prices[1], n_buy, day_prices[1], '%4d_b' %id]
                id += 1
                bought_days.append(day)
                #print('buy %0.2f' %float(n_buy*day_prices[1]))


        if day in np.array(bought_days) + 3:
            n = shop_dict['buy_%i' %(day - 3)][2]
            comp = shop_dict['buy_%i' %(day - 3)][0]
            id_s = shop_dict['buy_%i' %(day - 3)][4][:4]
            free_money += n*day_prices[0]
            shop_dict['sell_%i' %day] = [comp, n*day_prices[0], -n, day_prices[0], id_s + '_s']
            #print('sell %0.2f' %float(n*day_prices[0]))



    df = pd.DataFrame.from_dict(shop_dict, orient = 'index')
    df.columns = col_list
    df = df.sort_values('id')
    s = df['buy/sell'].values
    df['net'] = np.zeros(len(df))

    df.loc[1::2, 'net'] = s[1::2] + s[:-1:2]
    print(df)
    print('money made: %.1f' %(df['net'].values.sum()))
    print(free_money)

comps = ['Afarak Group', 'Affecto', 'Ahlstrom-Munksjö', 'Aktia Pankki A',
      'Alma Media', 'Amer Sports A', 'Apetit', 'Aspo',
      'Aspocomp Group', 'Atria A', 'Basware', 'Biohit B', 'Bittium',
      'CapMan', 'Cargotec', 'Caverion', 'Citycon', 'Cleantech Invest',
      'Componenta', 'Cramo', 'Digia',
      'Efore', 'Elisa', 'Endomines',
      'Etteplan', 'Exel Composites', 'F-Secure', 'Finnair',
      'Fiskars', 'Fortum', 'Glaston', 'HKScan A', 'Herantis Pharma',
      'Honkarakenne B', 'Huhtamäki',
      'Incap', 'Innofactor', 'Investors House', 'Kemira', 'Keskisuomalainen A',
      'Kesko A', 'Kesko B', 'Kone', 'Konecranes', 'Lassila & Tikanoja',
      'Lemminkäinen', 'Marimekko', 'Martela A', 'Metso', 'Metsä Board A',
      'Metsä Board B', 'Neo Industrial', 'Neste', 'Nokia', 'Nokian Renkaat',
      'Nordea Bank',  'OMXH25 ETF', 'Olvi A',
      'Orava Asuntorahasto', 'Oriola A', 'Oriola B', 'Orion A',
      'Orion B', 'Outokumpu', 'Outotec', 'Panostaja',
      'Ponsse', 'Pöyry', 'QPR Software',
      'Raisio K', 'Raisio V', 'Ramirent', 'Rapala VMC', 'Raute A',
      'Restamax', 'Revenio Group', 'SRV Yhtiöt', 'SSAB A', 'SSAB B',
      'SSH Comm. Security', 'Saga Furs C', 'Sampo A', 'Sanoma',
      'Scanfil', 'Siili Solutions', 'Solteq',
      'Sotkamo Silver', 'Sponda', 'Stockmann A', 'Stockmann B',
      'Stora Enso A', 'Stora Enso R', 'Suominen', 'Taaleri', 'Technopolis',
      'Teleste', 'Telia Company', 'Tieto', 'Tikkurila',
      'UPM-Kymmene', 'Uponor',
      'Uutechnic Group', 'Vaisala A', 'Valmet', 'Valoe', 'Verkkokauppa.com',
      'Viking Line', 'Wärtsilä', 'YIT',
      'eQ', 'Ålandsbanken B'] #


sum_ra = 0
threshold = 1.5 #%
comp_df_dict = {}
nfut = 2

for comp in comps[:40]:
    X_train, X_test, y_train, y_test, df = get_train_test(comp, [comp])
    y_pred, roc_auc = fit_predict(X_train, X_test, y_train, y_test, comp, [comp], threshold = threshold, plot_stock_b = False)
    if roc_auc > .6:
        df = df.loc[X_test.index.values + nfut][comp]
        df['prediction'] = y_pred
        comp_df_dict[comp] = df.iloc[::-1]
    #sum_ra += fit_Learner(comp, [comp], threshold, True )

test_simulate(comp_df_dict)

print('roc_auc average')
print(sum_ra/len(comps))
