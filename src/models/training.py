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
from sklearn.linear_model import ElasticNet
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor

from src.config import CONFIG
from src.models.gpu_data import (
    cudf_available,
    cuml_available,
    is_cuml_random_forest,
    make_cuml_random_forest,
    to_cudf_dataframe,
    to_cudf_series,
    xgboost_uses_cuda,
)
from src.models.registry import load_best_params
from src.utils.io import logger

ModelType = Union[RandomForestRegressor, XGBRegressor, LGBMRegressor, ElasticNet, LinearSVR]


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

    if model_name == "rf":
        if CONFIG.use_gpu and cuml_available():
            return make_cuml_random_forest(**params)
        params.setdefault("n_jobs", -1)
        return RandomForestRegressor(**params)
    if model_name == "xgb":
        params.setdefault("n_jobs", -1)
        params.setdefault("tree_method", "hist")
        params.setdefault("device", "cuda" if CONFIG.use_gpu else "cpu")
        return XGBRegressor(**params)
    if model_name == "lgbm":
        params.setdefault("n_jobs", -1)
        params.setdefault("device_type", "gpu" if CONFIG.use_gpu else "cpu")
        params.setdefault("verbosity", -1)
        return LGBMRegressor(**params)
    if model_name == "enet":
        return ElasticNet(**params)
    if model_name == "svr":
        return LinearSVR(**params)

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
    if isinstance(model, XGBRegressor) and xgboost_uses_cuda(model):
        if cudf_available():
            model.fit(to_cudf_dataframe(X_train), to_cudf_series(y_train))
            logger.info("XGBoost final training fitted with cuDF GPU data.")
        else:
            logger.warning("cuDF is unavailable; XGBoost final training is using pandas CPU data.")
            model.fit(X_train, y_train)
    elif is_cuml_random_forest(model):
        if cudf_available():
            model.fit(to_cudf_dataframe(X_train), to_cudf_series(y_train))
            logger.info("RandomForest final training fitted with cuML GPU backend.")
        else:
            model.fit(X_train.astype("float32"), y_train.astype("float32"))
            logger.info("RandomForest final training fitted with cuML (host arrays).")
    else:
        model.fit(X_train, y_train)
        if isinstance(model, RandomForestRegressor):
            logger.info("RandomForest final training fitted with sklearn CPU backend.")

    if isinstance(model, LGBMRegressor):
        logger.info(
            "LightGBM final training fitted with device_type=%s",
            model.booster_.params.get("device_type", "cpu"),
        )
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
