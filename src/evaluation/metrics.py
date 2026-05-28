#TODO: Persist the results table as parameters/metrics.csv for reproducibility.
"""
Performance metrics and model comparison.

Provides MAPE / RMSE / MAE / R^2 in the original (un-logged) target scale and
a tidy comparison DataFrame across models.
"""

########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from src.preprocessing.feature_engineering import inverse_log_transform_target

########################################################################################################################
#
# FUNCTIONS
#
########################################################################################################################
def _to_original_scale(y_log: np.ndarray | pd.Series) -> np.ndarray:
    """
    Undo the log1p target transformation used during training.
    """
    return inverse_log_transform_target(y_log)


def evaluate(model, X: pd.DataFrame, y_true_log: pd.Series) -> Dict[str, float]:
    """
    Score a single fitted model on the given split (target is log-scaled).
    """
    y_pred_log = model.predict(X)
    y_true = _to_original_scale(y_true_log)
    y_pred = _to_original_scale(y_pred_log)

    return {
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_all(models: Dict[str, object], X: pd.DataFrame, y_log: pd.Series) -> pd.DataFrame:
    """
    Apply evaluate() to every fitted model and return a tidy comparison DataFrame.
    Rows are model names; columns are metric names. Sorted by RMSE ascending.
    """
    rows = {name: evaluate(model, X, y_log) for name, model in models.items()}
    df = pd.DataFrame.from_dict(rows, orient="index")
    return df.sort_values("rmse", ascending=True)


def comparison_table(predictions_log: np.ndarray, y_true_log: pd.Series) -> pd.DataFrame:
    """
    Build a side-by-side Actual/Predicted DataFrame in the original target scale.
    """
    y_true = _to_original_scale(y_true_log)
    y_pred = _to_original_scale(predictions_log)
    return pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
