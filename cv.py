import model
import pandas as pd
import optuna
from enum import Enum

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import mean_squared_error

class CV(Enum):
    RF_STUDY = 'rf_optimization_new'
    XGB_STUDY = 'xgb_optimization'
    LGBM_STUDY = 'lgbm_optimization'
    STORAGE_URL = 'sqlite:///hyperparameter_opt.db'

def predict_test(
        n_days: int = 30, 
        days_back: int = 0,
        regressor: XGBRegressor | RandomForestRegressor | LGBMRegressor = XGBRegressor,
        params: dict = {},
        ) -> pd.DataFrame:
    
    preds = model.predict(
        n_days=n_days,
        days_back=days_back,
        training=True, 
        regressor=regressor,
        params=params
        )
    
    test = model.get_test_data(
        n_days=n_days, 
        days_back=days_back
        )

    return pd.merge(
        left=preds,
        right=test,
        left_index=True,
        right_index=True,
        suffixes=('_pred', '_test')
    )

def objective_xgb(trial):
    eta = trial.suggest_float('eta', 0.001, 1.0)
    subsample = trial.suggest_float('subsample', 0.001, 1.0)
    gamma = trial.suggest_float('gamma', 1e-5, 1_000_000, log=True)
    min_child_weight = trial.suggest_float('min_child_weight', 1e-5, 1_000_000, log=True)
    max_depth = trial.suggest_int('max_depth', 2, 15)

    params = {
        'booster': 'gbtree',
        'eta': eta,
        'subsample': subsample,
        'gamma': gamma,
        'min_child_weight': min_child_weight,
        'max_depth': max_depth
    }

    vals = predict_test(days_back=30, params=params)
    mse = mean_squared_error(vals['Num_pred'], vals['Num_test'])

    return mse

def objective_lgbm(trial):
    
    num_leaves = trial.suggest_int('num_leaves', 2, 256)
    learning_rate = trial.suggest_loguniform('learning_rate', 0.001, 0.1)
    feature_fraction = trial.suggest_uniform('feature_fraction', 0.1, 1.0)
    bagging_fraction = trial.suggest_uniform('bagging_fraction', 0.1, 1.0)
    bagging_freq = trial.suggest_int('bagging_freq', 1, 10)
    min_child_samples = trial.suggest_int('min_child_samples', 5, 100)
    reg_alpha = trial.suggest_loguniform('reg_alpha', 1e-8, 10.0)
    reg_lambda = trial.suggest_loguniform('reg_lambda', 1e-8, 10.0)
    
    params = {
        'num_leaves': num_leaves,
        'learning_rate': learning_rate,
        'feature_fraction': feature_fraction,
        'bagging_fraction': bagging_fraction,
        'bagging_freq': bagging_freq,
        'min_child_samples': min_child_samples,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda
    }

    vals = predict_test(days_back=30, params=params, regressor=LGBMRegressor)
    mse = mean_squared_error(vals['Num_pred'], vals['Num_test'])
    return mse

def objective_rf(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 2, 100)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features = trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2'])
    criterion = trial.suggest_categorical('criterion', ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'])

    params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_features': max_features,
        'criterion': criterion
    }

    vals = predict_test(days_back=30, params=params, regressor=RandomForestRegressor)

    mse = mean_squared_error(vals['Num_pred'], vals['Num_test'])

    return mse


def optimize_xgb(objective_xgb):
    storage_url = CV.STORAGE_URL.value
    study_name = CV.XGB_STUDY.value

    study = optuna.create_study(storage=storage_url, study_name=study_name, direction='minimize', load_if_exists=True)
    # study.optimize(objective_xgb, n_trials=100)
    print(study.best_params)

def optimize_lgbm(objective_lgbm):
    storage_url = CV.STORAGE_URL.value
    study_name = CV.LGBM_STUDY.value

    study = optuna.create_study(storage=storage_url, study_name=study_name, direction='minimize', load_if_exists=True)
    study.optimize(objective_lgbm, n_trials=100)
    # print(study.best_params)

def optimize_rf(objective_rf):
    storage_url = CV.STORAGE_URL.value
    study_name = CV.RF_STUDY.value

    study = optuna.create_study(storage=storage_url, study_name=study_name, direction='minimize', load_if_exists=True)
    print(study.get_trials())
    print(study.best_params)
    study.optimize(objective_rf, n_trials=100)

def get_best_model(
        storage_url: str = CV.STORAGE_URL.value, 
        studies: list[str] = [CV.LGBM_STUDY.value, CV.XGB_STUDY.value, CV.RF_STUDY.value],
        ) -> dict:
    """Gets the information on the best performing model.

    Args:
        storage_url (str, optional): Where the CV data are stored. Defaults to CV.STORAGE_URL.value.
        studies (list[str], optional): The different studies ran in CV. Defaults to [CV.LGBM_STUDY.value, CV.XGB_STUDY.value, CV.RF_STUDY.value].

    Returns:
        dict: A dictionary containing metadata on different studies.
    """
    
    params = list()
    for study in studies:
        data = dict()
        info = optuna.load_study(storage=storage_url, study_name=study)
        data['study'] = study
        data['score'] = info.best_value
        data['parameters'] = info.best_params

        params.append(data)

    return min(params, key=lambda x: x['score'])

if __name__ == '__main__':
    params = get_best_model()
    print(params)