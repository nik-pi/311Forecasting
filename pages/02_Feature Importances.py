from sklearn.ensemble import RandomForestRegressor
import pandas as pd


# import model.calldata

df = pd.read_csv('model/vals.csv')
# data = CallData(df)
# params = {'criterion': 'poisson', 'max_depth': 2, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 95}
# regressor = RandomForestRegressor(**params)