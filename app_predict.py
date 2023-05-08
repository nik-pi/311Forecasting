import model
from cv import get_best_model

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def get_regressor(
        info: dict
        ) -> LGBMRegressor | RandomForestRegressor | XGBRegressor:
    
    regressor = info.get('study')
    if regressor == 'xgb_optimization':
        return XGBRegressor
    elif regressor == 'lgbm_optimization':
        return LGBMRegressor
    else:
        return RandomForestRegressor

if __name__ == '__main__':
    # Get info on best model
    info = get_best_model()
    regressor = get_regressor(info=info)
    params = info.get('parameters')

    # Run Predictions
    preds = model.predict(
        days_back=0,
        training=False,
        regressor=regressor,
        params=params,
        n_days=30
    ).reset_index()

    # Save predictions
    df = pd.read_csv('model/preds.csv')
    preds = pd.concat([df, preds])
    df.to_csv('model/preds.csv', index=False)
    print(df)