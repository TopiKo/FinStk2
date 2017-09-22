import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, confusion_matrix
from sklearn.model_selection import GridSearchCV

def make_Xy(df, consider, predict, timeDelta = 5, aheadTime = 1, timeHistory = 365):
    #aheadTime #of days before we make prediction
    #timeDelta #of previous days to consider when making prediction

    df.reset_index(inplace = True)

    # Dual column indices:
    # Make new indexing as the df is going to be formatted heavily.
    # Rows timeDelta days will be transformed into columns with data and day labels.
    comps = np.reshape(np.array([np.repeat(comp[0].strip(), timeDelta) for comp in consider]), len(consider)*timeDelta)
    arr = np.reshape(np.array([['{:s}_{:d}'.format(val[1],-(i+aheadTime)) for i in range(timeDelta)] for val in consider]), timeDelta*len(consider))

    col_tuples = list(zip(*[comps, arr]))
    index = pd.MultiIndex.from_tuples(col_tuples, names=['company', 'values'])
    df_X = pd.DataFrame(columns = index)
    df_y = pd.Series(name = predict[0])

    for i in range(timeHistory):
        df_l = df.iloc[i + aheadTime : i + aheadTime + timeDelta][consider]
        data = np.reshape(df_l.values, timeDelta*len(consider), order = 'F')
        df_X.loc[i] = data
        df_y.loc[i] = df.iloc[i][predict].values.tolist()[0]

    #df_X.drop(predict[0][0], axis = 1, inplace = True)
    return df_X, df_y


def get_Xy(comp_predict, val_predict, excl_comp_list = [], timeDelta = 5, timeHistory = 365):

    df = pd.read_pickle('combined2.pkl')
    l1, l2 = zip(*df.columns.values)
    companies = list(set(l1))
    values = list(set(l2))

    # Predict with these:
    values_pick = ['Oe_price_change_%'] #'L_price_change_%'

    for comp in excl_comp_list:
        companies.remove(comp)

    companies_pick = np.repeat(np.array(sorted(companies)), len(values_pick))
    predict = [(comp_predict, val_predict)]

    consider = list(zip(*[companies_pick, values_pick*len(companies)]))
    dfX, dfy = make_Xy(df, consider, predict, timeDelta = timeDelta, timeHistory = timeHistory)

    return dfX, dfy


def fit_Learner():

    comp = 'Affecto'
    val = 'Oe_price_change_%'

    ndays = 6
    nhist = 365*4
    rem_comps = ['Ahlstrom-Munksjö', 'Ahola Transport A', 'Asiakastieto Group',
                 'Caverion', 'Cleantech Invest', 'Consti Yhtiöt', 'DNA',
                 'Detection Technology', 'Elite Varainhoito', 'Endomines',
                 'Evli Pankki', 'FIT Biotech', 'Fondia', 'Heeros',
                 'Herantis Pharma', 'Kamux Oyj', 'Kotipizza Group',
                 'Lehto Group', 'Nexstim', 'Next Games', 'Nixu',
                 'Orava Asuntorahasto', 'Pihlajalinna', 'Piippo',
                 'Privanet Group', 'Qt Group', 'Remedy Entertainment',
                 'Restamax', 'Robit', 'SSAB A', 'SSAB B', 'Savo-Solar',
                 'Scanfil', 'Siili Solutions', 'Silmäasema Oyj',
                 'Sotkamo Silver', 'Suomen Hoivatilat', 'Taaleri', 'Talenom',
                 'Talvivaara', 'Tokmanni Group', 'United Bankers', 'Valmet',
                 'Verkkokauppa.com', 'Vincit Group']

    X, y = get_Xy(comp, val, rem_comps, ndays, nhist)

    bad = []
    for col in X.columns.values.tolist():
        if 10 < sum(X[col].isnull()):
            if col[0] not in bad: bad.append(col[0])
            print('Xperse')

    if 10 < sum(y.isnull()):
        print('yperse')

    X.fillna(0, inplace = True)
    y.fillna(0, inplace = True)

    y =  y > .25
    X_train = X[int(nhist*.25):]
    y_train = y[int(nhist*.25):]
    X_test = X[:int(nhist*.25)]
    y_test = y[:int(nhist*.25)]



    rf = RandomForestClassifier(n_estimators = 1000, max_depth = 40, max_features = 200)
    param_grid = {'max_features': [100, 200],
                  'max_depth': [15, 20, 30],
                  'n_estimators': [500, 750, 1000]}

    #grid_rf = GridSearchCV(rf, param_grid, scoring='roc_auc', verbose = 3)
    grid_rf = rf
    grid_rf.fit(X_train, y_train)
    y_pred = grid_rf.predict(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr)

    #print(grid_rf.best_params_) #{'max_features': 200, 'n_estimators': 1000, 'max_depth': 30}
    #print(grid_rf.best_score_)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    plt.show()

fit_Learner()
