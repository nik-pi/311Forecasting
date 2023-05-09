import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from model import CallData

def get_feature_importances():
    df = pd.read_csv('model/vals.csv', parse_dates=['Date']).set_index('Date')
    data = CallData(df)
    
    X = data.X_train
    y = data.y_train

    model = RandomForestRegressor(
        criterion = 'poisson', 
        max_depth = 2, 
        max_features = 'log2', 
        min_samples_leaf = 1, 
        min_samples_split = 2, 
        n_estimators = 95
        )
    
    model.fit(X, y)
    feature_names = model.feature_names_in_
    importances = model.feature_importances_
    importances = zip(feature_names, importances)
    feature_importances = pd.DataFrame(importances, columns=['Feature', 'Importance'])

    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
    feature_importances.to_csv('pages/importances.csv', index=False)

if __name__ == '__main__':
    get_feature_importances()