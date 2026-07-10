import numpy as np


def wmae(y_true, y_pred, is_holiday):
    # kaggle metric, holiday weeks x5
    weights = np.where(is_holiday, 5, 1)
    return np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights)
