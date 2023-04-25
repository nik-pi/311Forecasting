import pandas as pd
import model

from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_percentage_error

if __name__ == '__main__':
    preds = model.predict(
        training=True,
        regressor=XGBRegressor,
        params={
            'eta': 0.5928255451426486, 
            'gamma': 0.00014479475888495706, 
            'max_depth': 3, 
            'min_child_weight': 2.084185190279089e-05, 
            'subsample': 0.9313420992192668
            },
        days_back=1
    )
    preds2 = model.predict(
        training=True,
        regressor=RandomForestRegressor,
        params={'criterion': 'poisson', 'max_depth': 2, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 95},
        days_back=1
    )
    test = model.get_test_data(days_back=1)
    df = model.combine_test_pred(test=test, preds=preds)
    df = pd.merge(left=df, right=preds2, left_index=True, right_index=True).rename(columns={'Num' : 'Num_rf'})
    df['% Error'] = (df['Num_pred'] - df['Num_test']) / df['Num_test']
    df['% Error'] = df['% Error']
    df['% Error RF'] = (df['Num_rf'] - df['Num_test']) / df['Num_test']
    df['% Error RF'] = df['% Error RF']
    df['% Error RF'] = df['% Error RF'].abs()
    df['% Error'] = df['% Error'].abs()
    import matplotlib.pyplot as plt
    plt.plot(df.index, df['% Error'])
    plt.plot(df.index, df['% Error RF'])

    from sklearn.metrics import mean_absolute_percentage_error
    mape_xgb = mean_absolute_percentage_error(df['Num_pred'], df['Num_test'])
    mape_rf = mean_absolute_percentage_error(df['Num_pred'], df['Num_rf'])
    print(f'{mape_xgb=}')
    print(f'{mape_rf=}')
    plt.show()

