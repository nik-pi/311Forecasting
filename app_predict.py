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
    # Run Predictions
    preds = model.predict(
        days_back=0,
        training=False,
        regressor=RandomForestRegressor,
        params={'criterion': 'poisson', 'max_depth': 2, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 95},
        n_days=7
    ).reset_index()

    # Save predictions
    df = pd.read_csv('model/preds.csv')
    preds = pd.concat([df, preds])
    preds.to_csv('model/preds.csv', index=False)