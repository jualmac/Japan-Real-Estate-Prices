"""
Train the final regression models using the best hyperparameters found by OptimizeRegressor.
"""
########################################################################################################################
#
# LIBRARIES
#
########################################################################################################################
from typing import Dict, Iterable, Union

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.config import CONFIG
from src.models.registry import load_best_params

ModelType = Union[RandomForestRegressor, XGBRegressor, LGBMRegressor]


########################################################################################################################
#
# FUNCTIONS
#
########################################################################################################################
def _instantiate(model_name: str, params: Dict) -> ModelType:
    """
    Build a fresh regressor with sensible defaults injected on top of the loaded params.
    """
    params = dict(params)
    params.setdefault("random_state", CONFIG.seed)
    params.setdefault("n_jobs", -1)

    if model_name == "rf":
        params.setdefault("verbose", -1)
        return RandomForestRegressor(**params)
    if model_name == "xgb":
        return XGBRegressor(**params)
    if model_name == "lgbm":
        params.setdefault("verbosity", -1)
        return LGBMRegressor(**params)

    raise ValueError(f"Unsupported model name: {model_name}")


def train_from_best_params(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> ModelType:
    """
    Load the best parameters from parameters/ and fit the matching regressor.
    """
    params = load_best_params(model_name)
    model = _instantiate(model_name, params)
    model.fit(X_train, y_train)
    return model


def train_all(
    model_names: Iterable[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Dict[str, ModelType]:
    """
    Fit every requested model and return a dict keyed by model name.
    """
    return {name: train_from_best_params(name, X_train, y_train) for name in model_names}
