import holidays
import numpy as np
import pandas as pd

class CallData:
    def __init__(
            self, 
            df: pd.DataFrame, 
            lags: list[int] = [1, 7, 14], 
            windows: list[int] = [5, 10, 15, 30, 45],
            ) -> None:
        
        self.df = df
        self.lags = lags
        self.windows = windows

        self._generate_features()

        self._train = self.df.iloc[:-1]
        self._pred = self.df.tail(1)

        self._train = self._train.dropna()
        self.X_train = self._train.drop(columns='Num')
        self.y_train = self._train['Num']
        self.X_pred = self._pred.drop(columns=['Num'])

    def _generate_features(
            self
            ) -> None:
        self.df = self.df.sort_index()
        self._add_prediction_row()

        self.df = self._add_date_features(self.df)
        self.df = self._add_holidays(self.df)
        self.df = self._calculate_lagged_features(self.df, 'Num', self.lags)
        self.df = self._generate_rolling_statistics(
            self.df, 
            col='Num', 
            windows=self.windows, 
            agg_funcs=['mean', 'std', 'median'], 
            split_by='WorkingDayWithHoliday'
            )
        
        self.df = self._ohe(
            self.df, 
            columns_to_encode=['Weekday', 'Quarter'],
            )
        
        self.df = self._calculate_cyclical_features(
            self.df, 
            cols=['Month', 'WeekOfYear', 'DayOfYear'],
            )

    def _add_prediction_row(
            self
            ) -> None:
        max_date_plus_one = self.df.index.max() + pd.Timedelta(days=1)
        self.df.loc[max_date_plus_one, 'Num'] = None

    def _get_holidays(
            self,
            df: pd.DataFrame
            ) -> pd.DataFrame:
        
        min_year, max_year = df.index.year.min(), df.index.year.max()
        holiday_dates = holidays.NYSE(observed=True, years=range(min_year, max_year+1))
        holiday_df = pd.DataFrame.from_dict(holiday_dates.items())
        holiday_df.columns = ['Date', 'Holiday']
        holiday_df['Date'] = pd.to_datetime(holiday_df['Date'])
        holiday_df = holiday_df.set_index('Date')

        return holiday_df

    def _add_holidays(
            self, 
            df:pd.DataFrame
            ) -> pd.DataFrame:
        
        holiday_df = self._get_holidays(df)
        holiday_df['Holiday3'] = 3
        holiday_df = holiday_df.asfreq('D')

        # Load Surrounding Values
        holiday_df['Holiday2'] = holiday_df['Holiday3'] - 1
        holiday_df['Holiday2'] = holiday_df['Holiday2']\
            .fillna(method='ffill', limit=1)\
            .fillna(method='bfill', limit=1)
        holiday_df['Holiday1'] = holiday_df['Holiday3'] - 2
        holiday_df['Holiday1'] = holiday_df['Holiday1']\
            .fillna(method='ffill', limit=2)\
            .fillna(method='bfill', limit=2)

        # Keep the max values and drop supplemental columns
        holiday_cols = ['Holiday3', 'Holiday2', 'Holiday1']
        holiday_df['Holiday'] = holiday_df[holiday_cols].max(axis=1).fillna(0)
        holiday_df = holiday_df.drop(columns=holiday_cols)

        df = pd.merge(
            left=df, 
            right=holiday_df, 
            left_index=True, 
            right_index=True
        )
        
        return df

    def _add_date_features(
            self, 
            df:pd.DataFrame
            ) -> pd.DataFrame:
        
        df['Weekday'] = df.index.weekday
        df['Month'] = df.index.month
        df['WeekOfYear'] = df.index.isocalendar().week.astype('int')
        df['DayOfYear'] = df.index.day
        df['Quarter'] = df.index.quarter
        
        holiday_df = self._get_holidays(df)
        df['WorkingDayWithHoliday'] = df.index.to_series().apply(
            lambda x: 1 if x.weekday() < 5 and x not in holiday_df.index else 0
            )
        
        return df
    
    def _generate_rolling_statistics(
            self,
            df: pd.DataFrame, 
            col: str,
            windows: list[int], 
            agg_funcs: list[str] = ['mean'], 
            split_by: str = None, 
            shift: int = 1
            ):

        if split_by:
            splits = df[split_by].unique()

        # Loop Over Each Window
        for window in windows:

            # Loop Over each Aggregation Method
            for agg_func in agg_funcs:

                # If data are split using a column
                if split_by:

                    # Needs a temporary DataFrame to store values
                    tempdf = pd.DataFrame()

                    # Loop over each split
                    for split in splits:
                        split = df[df[split_by] == split].copy()
                        split[f'Rolling_{split_by}_{window}_{agg_func}'] = split[col].shift(shift).rolling(window).agg(agg_func)

                        tempdf = pd.concat([tempdf, split])[[f'Rolling_{split_by}_{window}_{agg_func}']]
                
                # If not split by a column
                else:
                    df[f'Rolling_{split_by}_{window}_{agg_func}'] = df[col].shift(shift).rolling(window).agg(agg_func)

                # Recombine Data Column-wise
                df = pd.concat([df, tempdf], axis=1)

        return df


    def _calculate_lagged_features(
            self, 
            df:pd.DataFrame, 
            col:str,
            lags:list[int]
            ) -> pd.DataFrame:

        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        return df

    def _calculate_cyclical_features(
            self,
            df: pd.DataFrame, 
            cols: list[str],
            drop: bool = True
            ) -> pd.DataFrame:
        
        for col in cols:
            df[f'{col}_sin'] = np.sin(2 * np.pi * df[col]/df[col].max())
            df[f'{col}_cos'] = np.cos(2 * np.pi * df[col]/df[col].max())
        
        if drop:
            df = df.drop(columns=cols)
        
        return df
    
    def _ohe(
            self,
            df: pd.DataFrame, 
            columns_to_encode: list[str], 
            drop: bool = True
            ) -> pd.DataFrame:

        for col in columns_to_encode:
            df = pd.merge(
                left=df,
                right=pd.get_dummies(
                    df[col], 
                    prefix=col),
                left_index=True,
                right_index=True,
            )

        if drop:
            df = df.drop(columns=columns_to_encode)
            
        return df