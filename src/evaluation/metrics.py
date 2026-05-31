"""
Performance metrics and model comparison.

Provides RMSE / R^2 in log space plus RMSE / MAE / R^2 in the original Yen
target scale, returned as a tidy comparison DataFrame across models.
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
    mean_squared_error,
    r2_score,
)
from xgboost import XGBRegressor

from src.models.gpu_data import (
    cudf_available,
    is_cuml_random_forest,
    to_cudf_dataframe,
    to_host_array,
    xgboost_uses_cuda,
)
from src.preprocessing.feature_engineering import inverse_log_transform_target
from src.utils.io import logger

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


def predict_log(model, X: pd.DataFrame) -> np.ndarray:
    """
    Predict on the transformed target scale used during training.
    """
    if isinstance(model, XGBRegressor) and xgboost_uses_cuda(model):
        if cudf_available():
            y_pred_log = to_host_array(model.predict(to_cudf_dataframe(X)))
        else:
            logger.warning("cuDF is unavailable; XGBoost evaluation is using pandas CPU data.")
            y_pred_log = model.predict(X)
    elif is_cuml_random_forest(model):
        if cudf_available():
            y_pred_log = to_host_array(model.predict(to_cudf_dataframe(X)))
        else:
            y_pred_log = to_host_array(model.predict(X.astype("float32")))
    else:
        y_pred_log = model.predict(X)

    return np.asarray(y_pred_log)


def predict_original_scale(model, X: pd.DataFrame) -> np.ndarray:
    """
    Predict and invert the log1p target transformation back to the original scale.
    """
    return _to_original_scale(predict_log(model, X))


def evaluate(model, X: pd.DataFrame, y_true_log: pd.Series) -> Dict[str, float]:
    """
    Score a single fitted model on the given split.

    The target is log-scaled during training. Log-space metrics are useful for
    model comparison, while Yen-scale metrics are easier to interpret.
    """
    y_pred_log = predict_log(model, X)
    y_true_log_array = np.asarray(y_true_log)
    y_pred_yen = _to_original_scale(y_pred_log)
    y_true_yen = _to_original_scale(y_true_log_array)

    return {
        "rmse_log": float(np.sqrt(mean_squared_error(y_true_log_array, y_pred_log))),
        "r2_log": float(r2_score(y_true_log_array, y_pred_log)),
        "rmse_yen": float(np.sqrt(mean_squared_error(y_true_yen, y_pred_yen))),
        "mae_yen": float(mean_absolute_error(y_true_yen, y_pred_yen)),
        "r2_yen": float(r2_score(y_true_yen, y_pred_yen)),
    }


def evaluate_all(models: Dict[str, object], X: pd.DataFrame, y_log: pd.Series) -> pd.DataFrame:
    """
    Apply evaluate() to every fitted model and return a tidy comparison DataFrame.
    Rows are model names; columns are metric names. Sorted by Yen-scale RMSE ascending.
    """
    rows = {name: evaluate(model, X, y_log) for name, model in models.items()}
    df = pd.DataFrame.from_dict(rows, orient="index")
    return df.sort_values("rmse_yen", ascending=True)


def comparison_table(predictions_log: np.ndarray, y_true_log: pd.Series) -> pd.DataFrame:
    """
    Build a side-by-side Actual/Predicted DataFrame in the original target scale.
    """
    y_true = _to_original_scale(y_true_log)
    y_pred = _to_original_scale(predictions_log)
    return pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
