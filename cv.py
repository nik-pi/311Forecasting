import model
import pandas as pd
import optuna

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import mean_squared_error

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
    # booster = trial.suggest_categorical('booster', ('gbtree', 'gblinear'))
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

    vals = predict_test(days_back=1, params=params)

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

    vals = predict_test(days_back=1, params=params, regressor=RandomForestRegressor)

    mse = mean_squared_error(vals['Num_pred'], vals['Num_test'])

    return mse


def optimize_xgb(objective_xgb):
    storage_url = "sqlite:///hyperparameter_opt.db"
    study_name = "xgb_optimization"

    study = optuna.create_study(storage=storage_url, study_name=study_name, direction='minimize', load_if_exists=True)
    # study.optimize(objective_xgb, n_trials=100)
    print(study.best_params)

def optimize_rf(objective_rf):
    storage_url = "sqlite:///hyperparameter_opt.db"
    study_name = "rf_optimization_new"

    study = optuna.create_study(storage=storage_url, study_name=study_name, direction='minimize', load_if_exists=True)
    print(study.get_trials())
    print(study.best_params)
    # study.optimize(objective_rf, n_trials=100)
    # import optuna.visualization as vis

    # optimization_history_plot = vis.plot_optimization_history(study)
    # optimization_history_plot.show()
if __name__ == '__main__':
    optimize_rf(objective_rf)