import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

from .calldata import CallData

def _get_data(
        days_back : int = 0
    ) -> pd.DataFrame:
    return pd.read_csv(
        filepath_or_buffer='model/vals.csv', 
        dtype={'Num': int},
        parse_dates=['Date'],
        index_col='Date',
        skipfooter=days_back,
        engine='python'
        ).dropna()

def predict(
        n_days: int = 30,
        days_back: int = 0,
        training: bool = False,
        regressor = RandomForestRegressor,
        params: dict = {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 2, 'max_features': None}
    ):
    
    df = _get_data(days_back)
    
    if training:
        train = df.iloc[:-n_days].copy()
    else:
        train = df.copy()
    
    for _ in tqdm(range(n_days)):
        features = CallData(train)
        X_train = features.X_train 
        X_pred = features.X_pred
        y_train = features.y_train

        model = regressor(**params)
        model.fit(X_train, y_train)
        pred = model.predict(X_pred)

        pred_date = train.index.max() + pd.Timedelta(days=1)
        train.loc[pred_date, 'Num'] = pred

    train['Num'] = train['Num'].astype(int)
    preds = train[['Num']].tail(n_days)

    return preds

def get_test_data(
        n_days: int = 30,
        days_back: int = 0,
        ) -> pd.DataFrame:
    
    df = _get_data(days_back=days_back)
    return df.iloc[-n_days:].copy()

def combine_test_pred(
        test: pd.DataFrame,
        preds: pd.DataFrame
    ) -> pd.DataFrame:
    
    return pd.merge(
        left=preds,
        right=test,
        left_index=True,
        right_index=True,
        suffixes=('_pred', '_test')
    )