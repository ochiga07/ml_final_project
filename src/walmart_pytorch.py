import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, RegressorMixin

class WalmartPyTorchPipeline(BaseEstimator, RegressorMixin):
    # wrapper so we can save pytorch model + predict on raw test for mlflow
    # and handle the HORIZON=1 rolling forecast logic

    def __init__(self, model, lookback, device='cpu'):
        self.model = model
        self.lookback = lookback
        self.device = device
        self.train_raw_ = None

    def set_raw_data(self, train, features, stores):
        self.train_raw_ = train
        return self

    def fit(self, X, y=None):
        # We assume the model is already trained before being passed to this wrapper
        # This wrapper is primarily for inference and MLflow tracking
        return self

    def _predict_one_week(self, history, week_test):
        self.model.eval()
        preds = []
        
        # history has ['Store', 'Dept', 'Date', 'Weekly_Sales']
        # week_test has ['Store', 'Dept', 'Date']
        
        for _, row in week_test.iterrows():
            store = row['Store']
            dept = row['Dept']
            
            h = history[(history['Store'] == store) & (history['Dept'] == dept)]
            
            if len(h) < self.lookback:
                # pad with zeros if not enough history
                sales = np.zeros(self.lookback, dtype=np.float32)
                if len(h) > 0:
                    sales[-len(h):] = h['Weekly_Sales'].values
            else:
                sales = h['Weekly_Sales'].values[-self.lookback:].astype(np.float32)
                
            x_tensor = torch.FloatTensor(sales).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred = self.model(x_tensor).cpu().numpy().flatten()[0]
                
            # clip negative predictions to 0
            preds.append(max(0.0, pred))
            
        return np.array(preds)

    def predict(self, X):
        if self.train_raw_ is None:
            raise ValueError('Training data not saved on model. Call set_raw_data() first.')

        # X is the raw test dataframe
        test = X[['Store', 'Dept', 'Date']].copy()
        test['_row_id'] = np.arange(len(test))

        history = self.train_raw_[['Store', 'Dept', 'Date', 'Weekly_Sales']].copy()
        result_parts = []

        # predict week by week so 1-step forecast rolls forward
        for date in sorted(test['Date'].unique()):
            week_test = test[test['Date'] == date].copy()
            preds = self._predict_one_week(history, week_test)

            week_out = week_test[['_row_id']].copy()
            week_out['prediction'] = preds
            result_parts.append(week_out)

            week_history = week_test[['Store', 'Dept', 'Date']].copy()
            week_history['Weekly_Sales'] = preds
            history = pd.concat([history, week_history], ignore_index=True)

        out = pd.concat(result_parts, ignore_index=True)
        out = out.sort_values('_row_id')
        return out['prediction'].values
