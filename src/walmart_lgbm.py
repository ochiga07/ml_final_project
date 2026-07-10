import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.base import BaseEstimator, RegressorMixin

from feature_pipeline import run_feature_pipeline


def prepare_xy(df, feature_cols):
    X = df[feature_cols].copy()
    for col in ['Store', 'Dept', 'Type']:
        if col in X.columns:
            X[col] = X[col].astype('category')
    return X


class WalmartLGBMPipeline(BaseEstimator, RegressorMixin):
    # wrapper so we can save model + predict on raw test for mlflow

    def __init__(self, feature_cols, lgb_params=None):
        self.feature_cols = feature_cols
        self.lgb_params = lgb_params or {
            'objective': 'regression_l1',
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'verbose': -1,
        }
        self.train_raw_ = None
        self.features_ = None
        self.stores_ = None
        self.model_ = None

    def set_raw_data(self, train, features, stores):
        self.train_raw_ = train
        self.features_ = features
        self.stores_ = stores
        return self

    def fit(self, X, y):
        self.model_ = lgb.LGBMRegressor(**self.lgb_params)
        self.model_.fit(X, y)
        return self

    def _predict_one_week(self, history, week_test):
        week_input = week_test[['Store', 'Dept', 'Date', 'IsHoliday']].copy()
        full_df = run_feature_pipeline(history, week_input, self.features_, self.stores_)
        week_df = full_df[full_df['is_train'] == 0].drop(columns=['is_train'])
        X = prepare_xy(week_df, self.feature_cols)
        return np.clip(self.model_.predict(X), 0, None)

    def predict(self, X):
        if 'lag_1' in X.columns:
            X = prepare_xy(X[self.feature_cols], self.feature_cols)
            return np.clip(self.model_.predict(X), 0, None)

        if self.train_raw_ is None:
            raise ValueError('Training data not saved on model. Call set_raw_data() first.')

        test = X[['Store', 'Dept', 'Date', 'IsHoliday']].copy()
        test['_row_id'] = np.arange(len(test))

        history = self.train_raw_[['Store', 'Dept', 'Date', 'IsHoliday', 'Weekly_Sales']].copy()
        result_parts = []

        # predict week by week so lags work on test
        for date in sorted(test['Date'].unique()):
            week_test = test[test['Date'] == date].copy()
            preds = self._predict_one_week(history, week_test)

            week_out = week_test[['_row_id']].copy()
            week_out['prediction'] = preds
            result_parts.append(week_out)

            week_history = week_test[['Store', 'Dept', 'Date', 'IsHoliday']].copy()
            week_history['Weekly_Sales'] = preds
            history = pd.concat([history, week_history], ignore_index=True)

        out = pd.concat(result_parts, ignore_index=True)
        out = out.sort_values('_row_id')
        return out['prediction'].values
